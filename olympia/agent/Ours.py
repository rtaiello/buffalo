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

LMBDA = 2048


class OursClientAgent(AggregationClient):

    def __init__(self, id, name, type, *args, **kwargs):
        super().__init__(id, name, type, *args, **kwargs)
        self.rounds_to_skip = [1]
        self.rounds_to_add = {2: {1e5: 6627947807, 1e6: 67024679422}}

    def round(self, round_number, message):
        if round_number == 1:

            ################ INITIALIZE #################
            self.GF = self.params["gf"]
            self.random_state = self.params["random_state"]
            self.num_clients = self.params["num_clients"]
            self.s_len = self.params["s_len"]
            self.dim = self.params["dim"]
            ################ RLWE ################
            if self.params["dim"] >= 1e5:
                path_A = (
                    f"/home/argentera/taiello/olympia/cache/A_{self.dim}.npy"
                )
                path_A_s = (
                    f"/home/argentera/taiello/olympia/cache/A_s_{self.dim}.npy"
                )
                if not os.path.exists(path_A):
                    A = make_A(
                        self.params["dim"],
                        self.s_len,
                        self.GF,
                        message["random_number"],
                    )
                    ones = self.GF.Ones(self.s_len)
                    t1 = time.time()
                    A_s = A.dot(ones)
                    t2 = time.time()
                    ct = int((t2 - t1) * 1e9)
                    print(f"Time to compute A_s: {ct}")
                    np.save(path_A, A)
                    np.save(path_A_s, A_s)
            else:
                self.A = make_A(
                    self.params["dim"], self.s_len, self.GF, message["random_number"]
                )

            ################ THRESHOLD PAILLIER ################
            self.sk_share = message["sk_share"]  # private key share
            self.pk_paillier = message["pk"]  # public key
            self.ves = VES(LMBDA // 2, self.params["num_clients"], 32, self.s_len)
            self.T = int(self.num_clients * self.params["frac_honests"]) + 1
            self.tpaillier = TPHE(self.T, self.num_clients)
            ################ JOYE LIBERT ################
            self.jls = JLS(self.num_clients, self.ves)
            self.pp, _, _ = self.jls.setup(lmbda=LMBDA)
            ################ SIGNING KEY ################
            self.sign_u = SigningKey.generate()
            self.verify_u = self.sign_u.verify_key
            return self.verify_u

        if round_number == 2:

            self.verify_keys = message
            self.s = self.make_s(self.s_len)  # generate keys

            # Matrix A it should be generated at round 0 but for memory efficiency we generate it here
            if self.params["dim"] >= 1e5:
                A_s = self.GF(
                    np.load(
                        f"/home/argentera/taiello/olympia/cache/A_s_{self.dim}.npy"
                    )
                )
                masked_value = self.mask_message(A_s, ones=True)
            else:
                masked_value = self.mask_message(self.A)
            self.pks = message
            s_list = self.s.tolist()
            seed: random.Random = random.SystemRandom()
            key_len = self.pp.n.bit_length() - int(log2(self.num_clients))
            u_k = seed.getrandbits(2 * key_len)
            user_key = UserKeyJL(self.pp, u_k)
            enc_s_jl = self.jls.protect(self.pp, user_key, 0, s_list)
            enc_u_phe = self.tpaillier.encrypt(self.pk_paillier, u_k)
            self.enc_u_phe = [enc_u_phe]
            return {"enc_s_jl": enc_s_jl, "enc_u_phe": enc_u_phe, "masked_value": masked_value}

        if round_number == 3:

            self.signed = self.sign_u.sign(message)
            return {"signed": self.signed}

        if round_number == 4:

            for key in message.keys():
                if message[key]["signed"]:
                    self.verify_keys[key].verify(message[key]["signed"])
                    assert self.signed.message == message[key]["signed"].message
            self.enc_u_phe.extend([message[key]["enc_u_phe"] for key in message.keys()])
            enc_sum_u_phe = self.tpaillier.sum(self.enc_u_phe)
            partial_dec_u_phe = self.tpaillier.partial_decrypt(self.sk_share, enc_sum_u_phe)
            return partial_dec_u_phe

    def mask_message(self, A, ones=False):
        client_value = self.GF.Ones(self.dim)
        if ones:
            return client_value + A

        return client_value + A.dot(self.s)

    def make_s(self, n, ones=True):
        if ones:
            return self.GF.Ones(n)
        return self.GF.Random(n)


class OursServiceAgent(DropoutAggregationServer):

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
            n = len(self.clients)
            self.T = int(n * self.params["frac_honests"]) + 1
            self.s_len = self.params["s_len"]
            random_number = int(self.GF.Random(()))
            ################ RLWE ################
            if self.params["dim"] < 1e5:
                self.A = make_A(self.params["dim"], self.s_len, self.GF, random_number)
            ################ JOYE LIBERT ################
            self.ves = VES(LMBDA // 2, n, 32, self.s_len)
            self.jls = JLS(n, self.ves)
            self.pp, _, _ = self.jls.setup(lmbda=LMBDA)
            ################ THRESHOLD PAILLIER ################
            self.tpaillier = TPHE(self.T, n)
            self.pk, sk = self.tpaillier.setup(lmbda=LMBDA*2)
            sk_shares = self.tpaillier.sk_share(sk)

            return {
                client: {
                    "random_number": random_number,
                    "pk": self.pk,
                    "sk_share": sk_shares[idx],
                }
                for idx, client in enumerate(self.clients)
            }

        if round_number == 2:
            verify_keys = messages
            return {client: verify_keys for client in verify_keys.keys()}

        if round_number == 3:
            self.enc_s_jl = defaultdict(dict)
            self.enc_u_phe = defaultdict(dict)
            self.masked_values = defaultdict(dict)
            self.u3 = defaultdict(dict)
            for source in messages.keys():
                self.masked_values[source] = messages[source]["masked_value"]
                self.enc_s_jl[source] = messages[source]["enc_s_jl"]
                self.enc_u_phe[source] = messages[source]["enc_u_phe"]
            return {client: dill.dumps(messages.keys()) for client in messages.keys()}

        if round_number == 4:
            out_messages = defaultdict(dict)
            for dest in messages.keys():
                for source in self.enc_s_jl.keys():
                    if source == dest:
                        continue
                    signed = (
                        messages[source]["signed"]
                        if source in messages.keys()
                        else None
                    )
                    out_messages[dest][source] = {
                        "enc_u_phe": self.enc_u_phe[source],
                        "signed": signed,
                    }
            return out_messages

        if round_number == 5:
            u4 = set(messages.keys()).union(set(self.masked_values.keys()))
            mask_vals = self.GF(
                [
                    self.masked_values[k]
                    for k in u4
                    if isinstance(self.masked_values[k], self.GF)
                ]
            )

            self.total = mask_vals.sum(axis=0)
            partial_decs_u_phe = []
            for message in messages.values():
                partial_decs_u_phe.append(message)
            partial_decs_u_phe = partial_decs_u_phe[: self.T]
            u_k_0 = self.tpaillier.decrypt(self.pk, partial_decs_u_phe)
            server_key = ServerKeyJL(self.pp, -u_k_0)
            s_server = self.jls.agg(self.pp, server_key, 0, list(self.enc_s_jl.values()))
            self.s = np.mod(s_server, self.GF.order)
            if self.params["dim"] < 1e5:
                A = self.A
            else:
                A = self.GF(
                    np.load(
                        f"/home/argentera/taiello/olympia/cache/A_{self.params['dim']}.npy"
                    )
                )
            self.aggregate = self.total - A.dot(self.GF(self.s[: self.s_len]))
            self.succeed(result=self.aggregate)


def make_A(vec_size, n, GF, seed):
    return GF.Random((vec_size, n), seed=seed)
