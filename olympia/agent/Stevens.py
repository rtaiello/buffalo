import os
import time
from collections import defaultdict

import dill
import numpy as np
from nacl.public import Box, PrivateKey
from nacl.signing import SigningKey

import util.shamir_sharing as shamir
from agent.AggregationAgent import AggregationClient, DropoutAggregationServer
from util import util
from util.util import log_print
from rlwe_sa import RlweSA, VES
from math import log2


class StevensClientAgent(AggregationClient):

    def __init__(self, id, name, type, *args, **kwargs):
        super().__init__(id, name, type, *args, **kwargs)
        self.rounds_to_skip = [1]
        self.rounds_to_add = {}

    def round(self, round_number, message):
        if round_number == 1:

            ################ INITIALIZE #################
            self.GF = self.params["gf"]
            self.random_state = self.params["random_state"]
            self.s_len = self.params["s_len"]
            self.dim = self.params["dim"]
            self.num_clients = self.params["num_clients"]
            ################ RLWE ################            
            seed = message
            ptxt_size = int(58 - log2(self.num_clients))
            value_size = 8
            self.ves_rlwe = VES(ptxt_size, self.num_clients, value_size, self.dim)
            self.new_dim = self.dim // self.ves_rlwe.compratio
            self.rlwe = RlweSA(self.new_dim, ptxt_size, seed)
            self.sk_u = PrivateKey.generate()
            self.pk_u = self.sk_u.public_key

            ################ SIGNING KEY ################
            self.sign_u = SigningKey.generate()
            self.verify_u = self.sign_u.verify_key
            return {"pk_u": self.pk_u, "verify_u": self.verify_u}

        if round_number == 2:
            still_alive = list(sorted(message.keys()))
            n = len(still_alive)

            self.s = self.rlwe.gen_secret_key()
            plaintext = [1] * self.dim
            plaintext_enc = self.ves_rlwe.encode(plaintext)
            masked_value = self.rlwe.encrypt(self.s, plaintext_enc)
            self.T = int(n * self.params["frac_honests"])
            dropouts_per_round = int(n * self.params["dropout_fraction"])
            expected_dropouts = dropouts_per_round * 4
            if self.params["packing"]:
                self.K = n - self.T - expected_dropouts - 1
            else:
                self.K = 1
            self.pks = message
            s_list = self.rlwe.key_to_vector(self.s)
            self.s = np.array(s_list,dtype=np.int64)
            shares = shamir.share_array(self.s, still_alive, self.T, self.GF, K=self.K)
            enc_shares = {
                c: shares[c].encrypt(self.sk_u, pk["pk_u"])
                for c, pk in self.pks.items()
                if c in shares
            }
            return {"enc_shares": enc_shares, "masked_value": masked_value}

        if round_number == 3:
            self.signed = self.sign_u.sign(message)
            return {"signed": self.signed}

        if round_number == 4:
            for key in message.keys():
                if message[key]["signed"]:
                    self.pks[key]["verify_u"].verify(message[key]["signed"])
                    assert self.signed.message == message[key]["signed"].message

            dec_shares = [
                s["out_shares"].decrypt(self.sk_u, self.pks[c]["pk_u"])
                for c, s in message.items()
                if c in self.pks
            ]
            return shamir.sum_share_array(dec_shares)


class StevensServiceAgent(DropoutAggregationServer):

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
            self.GF = self.params["gf"]
            self.s_len = self.params["s_len"]
            self.threshold = len(self.clients)
            self.dim = self.params["dim"]
            self.num_clients = self.params["num_clients"]
            ################ RLWE ################
            
            ptxt_size = int(58 - log2(self.num_clients))
            value_size = 8
            self.ves_rlwe = VES(ptxt_size, self.num_clients, value_size, self.dim)
            self.new_dim = self.dim // self.ves_rlwe.compratio
            self.rlwe = RlweSA(self.new_dim, ptxt_size)
            seed = self.rlwe.get_seed()
            return {client: seed for client in self.clients}

        if round_number == 2:
            all_pks_verifys = {client: messages for client in messages.keys()}
            return all_pks_verifys

        if round_number == 3:
            self.out_shares = defaultdict(dict)
            self.masked_values = defaultdict(dict)
            for source in messages.keys():
                self.masked_values[source] = messages[source]["masked_value"]
                for dest in messages[source]["enc_shares"].keys():
                    self.out_shares[dest][source] = messages[source]["enc_shares"][dest]
            return {client: dill.dumps(messages.keys()) for client in messages.keys()}

        if round_number == 4:
            out_shares = defaultdict(dict)
            for dest in messages.keys():
                for source in self.out_shares[dest].keys():
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
            self.shares = messages
            u4 = set(self.shares.keys()).union(set(self.masked_values.keys()))
            mask_vals = [self.masked_values[k] for k in u4]
            chipertext_sum = mask_vals[0]
            for chipertext in mask_vals[1:]:
                chipertext_sum = self.rlwe._rlwe_sa.aggregate(chipertext_sum, chipertext)
            self.total = chipertext_sum
            s_server= shamir.reconstruct_array(list(self.shares.values()))
            s_server = s_server.tolist()[:self.s_len]
            self.s = [val% self.rlwe.get_modulus_key for val in s_server]
            self.s = self.rlwe.vector_to_key(self.s)
            aggregate_enc = self.rlwe.decrypt(self.s, self.total) 
            self.aggregate = self.ves_rlwe.decode(aggregate_enc)
            self.succeed(result=self.aggregate)


def make_A(vec_size, n, GF, seed):
    return GF.Random((vec_size, n), seed=seed)
