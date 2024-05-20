import os
import time
from collections import defaultdict
import random

import dill
import numpy as np
from nacl.signing import SigningKey

from agent.AggregationAgent import AggregationClient, DropoutAggregationServer
from util import util
from util.paillier.tpaillier.tphe import TPHE
from util.joye_libert.vector_encoding import VES
from util.util import log_print
from util.joye_libert.jl import JLS
from util.joye_libert.jl_utils import UserKeyJL, ServerKeyJL
from math import log2
from util.ss.shamir_ss import SSS
from rlwe_sa.shell_encryption import RlweSA

LMBDA = 2048


class OursSSClientAgent(AggregationClient):

    def __init__(self, id, name, type, *args, **kwargs):
        super().__init__(id, name, type, *args, **kwargs)
        self.rounds_to_skip = [1]
        self.rounds_to_add = {}

    def round(self, round_number, message):
        if round_number == 1:

            ################ INITIALIZE #################
            self.random_state = self.params["random_state"]
            self.num_clients = self.params["num_clients"]
            self.s_len = self.params["s_len"]
            self.dim = self.params["dim"]
            
            ################ RLWE ################
            
            seed = message
            ptxt_size = int(58 - log2(self.num_clients))
            value_size = 8
            self.ves_rlwe = VES(ptxt_size, self.num_clients, value_size, self.dim)
            self.new_dim = self.dim // self.ves_rlwe.compratio
            self.rlwe = RlweSA(self.new_dim, ptxt_size, seed)
            ################ SHAMIR SECRET SHARING ################
            self.ss = SSS(LMBDA)
            self.T = int(self.num_clients * self.params["frac_honests"]) + 1
            self.enc_shares = []
            ################ JOYE LIBERT ################
            self.ves = VES(LMBDA // 2,  self.num_clients, 59, self.s_len)
            self.jls = JLS(self.num_clients, self.ves)
            self.pp, _, _ = self.jls.setup(lmbda=LMBDA)
            ################ SIGNING KEY ################
            self.sign_u = SigningKey.generate()
            self.verify_u = self.sign_u.verify_key
            return self.verify_u

        if round_number == 2:

            self.verify_keys = message
            self.s = self.rlwe.gen_secret_key()
            plaintext = [1] * self.dim
            plaintext_enc = self.ves_rlwe.encode(plaintext)
            masked_value = self.rlwe.encrypt(self.s, plaintext_enc)
            self.pks = message
            s_list = self.rlwe.key_to_vector(self.s)
            seed: random.Random = random.SystemRandom()
            key_len = self.pp.n.bit_length() - int(log2(self.num_clients))
            u_k = seed.getrandbits(2 * key_len)
            user_key = UserKeyJL(self.pp, u_k)
            enc_s_jl = self.jls.protect(self.pp, user_key, 0, s_list)
            shares_sk = self.ss.share(u_k, self.T, self.num_clients)
            self.enc_shares.append(shares_sk[self.id])
            return {"enc_s_jl": enc_s_jl, "enc_shares": shares_sk, "masked_value": masked_value}

        if round_number == 3:

            self.signed = self.sign_u.sign(message)
            return {"signed": self.signed}

        if round_number == 4:
            for key in message.keys():
                if message[key]["signed"]:
                    self.verify_keys[key].verify(message[key]["signed"])
                    assert self.signed.message == message[key]["signed"].message
            self.enc_shares.extend([message[key]["out_shares"] for key in message.keys()])
            
            sk_0_share = self.enc_shares[0]
            for share in self.enc_shares[1:]:
                sk_0_share += share
            return sk_0_share


    def make_s(self, n, ones=True):
        if ones:
            return self.GF.Ones(n)
        return self.GF.Random(n)


class OursSSServiceAgent(DropoutAggregationServer):

    def __init__(
        self, id, name, type, client_ids, params, random_state, *args, **kwargs
    ):
        super().__init__(
            id, name, type, client_ids, params, random_state, *args, **kwargs
        )
        self.rounds_to_skip = [1, 2]
        self.dropout_fraction = self.params["dropout_fraction"]
        self.T = int(self.params["num_clients"] * self.params["frac_honests"])

    def round(self, round_number, messages):

        if round_number == 1:
            ################ INITIALIZE #################
            self.GF = self.params["gf"]
            self.num_clients = self.params["num_clients"]
            self.dim = self.params["dim"]
            self.T = int(self.num_clients * self.params["frac_honests"]) + 1
            self.s_len = self.params["s_len"]
            ################ RLWE ################
            
            ptxt_size = int(58 - log2(self.num_clients))
            value_size = 8
            self.ves_rlwe = VES(ptxt_size, self.num_clients, value_size, self.dim)
            self.new_dim = self.dim // self.ves_rlwe.compratio
            self.rlwe = RlweSA(self.new_dim, ptxt_size)
            seed = self.rlwe.get_seed()
            ################ JOYE LIBERT ################
            self.ves = VES(LMBDA // 2, self.num_clients, 59, self.s_len)
            self.jls = JLS(self.num_clients, self.ves)
            self.pp, _, _ = self.jls.setup(lmbda=LMBDA)
            ################ SHAMIR SECRET SHARING  ################
            self.ss = SSS(LMBDA)

            return {client: seed for client in self.clients}


        if round_number == 2:
            verify_keys = messages
            return {client: verify_keys for client in verify_keys.keys()}

        if round_number == 3:
            self.enc_s_jl = defaultdict(dict)
            self.out_shares = defaultdict(dict)
            self.masked_values = defaultdict(dict)
            self.u3 = defaultdict(dict)
            for source in messages.keys():
                self.masked_values[source] = messages[source]["masked_value"]
                self.enc_s_jl[source] = messages[source]["enc_s_jl"]

                for dest in messages[source]["enc_shares"].keys():
                    if source == dest:
                        continue
                    self.out_shares[dest][source] = messages[source]["enc_shares"][dest]
            return {client: dill.dumps(messages.keys()) for client in messages.keys()}

        if round_number == 4:
            out_shares = defaultdict(dict)
            for dest in messages.keys():
                for source in self.out_shares.keys():
                    if source == dest:
                        continue
                    signed = (
                        messages[source]["signed"]
                        if source in messages.keys()
                        else None
                    )
                    out_shares[dest][source] = {
                        "out_shares": self.out_shares[dest][source],
                        "signed": signed,
                    }

            return out_shares

        if round_number == 5:
            u4 = set(messages.keys()).union(set(self.masked_values.keys()))
            mask_vals = [self.masked_values[k] for k in u4]
            chipertext_sum = mask_vals[0]
            for chipertext in mask_vals[1:]:
                chipertext_sum = self.rlwe._rlwe_sa.aggregate(chipertext_sum, chipertext)
            self.total = chipertext_sum
            sk_0_shares = []
            for message in messages.values():
                sk_0_shares.append(message)
            sk_0_shares = sk_0_shares[: self.T]
            lag_coeffs = self.ss.lagrange(sk_0_shares)
            u_k_0 = self.ss.reconstruct(sk_0_shares, self.T, lag_coeffs)
            server_key = ServerKeyJL(self.pp, -u_k_0)
            s_server = self.jls.agg(self.pp, server_key, 0, list(self.enc_s_jl.values()))
            self.s = [val% self.rlwe.get_modulus_key for val in s_server]
            self.s = self.rlwe.vector_to_key(self.s)
            aggregate_enc = self.rlwe.decrypt(self.s, self.total) 
            self.aggregate = self.ves_rlwe.decode(aggregate_enc)
            self.succeed(result=self.aggregate)
