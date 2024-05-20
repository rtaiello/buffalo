import os
import time
from collections import defaultdict

import dill
import numpy as np
from Crypto.PublicKey import ECC
from Crypto.PublicKey.ECC import EccPoint
from nacl.public import Box, PrivateKey
from nacl.signing import SigningKey

import util.shamir_sharing as shamir
from agent.AggregationAgent import AggregationClient, DropoutAggregationServer
from util import util
from util.lcc_ec import LCCEllipticCurve
from util.util import log_print
from util.elgamal.teg import TElGamal

curve_name = "P-256"
_curve = ECC._curves[curve_name]



class OursVeriSAClientAgent(AggregationClient):

    def __init__(self, id, name, type, *args, **kwargs):
        super().__init__(id, name, type, *args, **kwargs)
        self.rounds_to_skip = [1]
        self.rounds_to_add = None

    def round(self, round_number, message):
        if round_number == 1:


            ################ INITIALIZE #################
            self.random_state = self.params["random_state"]
            self.dim = self.params["dim"]

            ################ ECC ################
            self.lcc = LCCEllipticCurve(_curve)
            self.order = _curve.order

            self.teg = TElGamal(self.params["num_clients"], self.params["num_clients"])
            self.distinct_bases = message

            self.sk_u = PrivateKey.generate()
            self.pk_u = self.sk_u.public_key

            ################ SIGNING KEY ################
            self.sign_u = SigningKey.generate()
            self.verify_u = self.sign_u.verify_key
            self.distinct_bases = message["distict_bases"]
            self.pk = message["pk"]
            self.sk_share = message["sk_share"]
            return {"pk_u": self.pk_u, "verify_u": self.verify_u}

        if round_number == 2:
            
            self.pks = message
            
            noise = self.lcc._generate_point_ec()
            
            masked_value = np.ones(self.dim, dtype=np.uint8)
            hash = generate_hash(masked_value, self.distinct_bases, self.dim)

            commitment, self.pedersen_g, self.pedersen_l = self.lcc.pedersen_commitment(
                hash, noise
            )
            enc_hash = self.teg.encrypt(hash, self.pk)
            return {
                "enc_hash": enc_hash,
                "masked_value": masked_value,
                "commitment": commitment,
                "noise": noise,
            }

        if round_number == 3:
            self.signed = self.sign_u.sign(message)
            return {"signed": self.signed}

        if round_number == 4:

            for key in message.keys():
                if message[key]["signed"]:
                    self.pks[key]["verify_u"].verify(message[key]["signed"])
                    assert self.signed.message == message[key]["signed"].message

            enc_hashes = [s["enc_hash"] for c, s in message.items() if c in self.pks]
            self.noises = [s["noise"] for c, s in message.items()]
            self.commitments = [s["commitment"] for c, s in message.items()]
            enc_sum = enc_hashes[0]
            for i in range(1, len(enc_hashes)):
                enc_sum = enc_sum + enc_hashes[i]
            partial_dec  = self.teg.share_decrypt(enc_sum, self.sk_share)
            return {"partial_dec": partial_dec, "c2": enc_sum.c2}

        if round_number == 5:
            hash_sum = message["hash_sum"]
            agg_value = message["agg_value"]
            commitment_sum = self.commitments[0]
            noise_sum = self.noises[0]
            n = len(self.commitments)
            for i in range(1, n):
                commitment_sum = commitment_sum + self.commitments[i]
                noise_sum = noise_sum + self.noises[i]
            commitment_sum = commitment_sum + (
                (self.order - n + 1) * (self.pedersen_g + self.pedersen_l)
            )
            commitment_agg, _, _ = self.lcc.pedersen_commitment(hash_sum, noise_sum)
            assert commitment_agg == commitment_sum
            has_agg = generate_hash(agg_value, self.distinct_bases, self.dim)
            assert has_agg == hash_sum

            return True


def generate_hash(masked_value, distinct_bases, dim):
    hash_client = masked_value[0] * distinct_bases[0]
    for i in range(1, dim):
        temp_hash = masked_value[i] * distinct_bases[i]
        hash_client = hash_client + temp_hash
    return hash_client


class OursVeriSAServiceAgent(DropoutAggregationServer):

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
            _curve = ECC._curves[curve_name]
            self.P = ECC.EccPoint(x=_curve.Gx, y=_curve.Gy, curve=curve_name)
            self.lcc_ec = LCCEllipticCurve(_curve)
            self.random_state = self.params["random_state"]
            alphas = self.random_state.randint(low=0, high=100, size=self.params["dim"])
            distinct_bases = self.distinct_points_compute(alphas)
            self.T = int(self.params["num_clients"] * self.params["frac_honests"])
            self.n = len(self.clients)
            self.teg = TElGamal(self.T, self.n)
            self.pk, self.sk_shares = self.teg.setup()
            return {
                client: {
                    "distict_bases": distinct_bases,
                    "pk": self.pk,
                    "sk_share": self.sk_shares[idx],
                }
                for idx, client in enumerate(self.clients)
            }

        if round_number == 2:
            all_pks_verifys = {client: messages for client in messages.keys()}
            return all_pks_verifys

        if round_number == 3:
            self.enc_hashes = defaultdict(dict)
            self.noises = defaultdict(dict)
            self.commitements = defaultdict(dict)
            self.masked_values = defaultdict(dict)
            for source in messages.keys():
                self.masked_values[source] = messages[source]["masked_value"]
                self.noises[source] = messages[source]["noise"]
                self.commitements[source] = messages[source]["commitment"]
                self.enc_hashes[source] = messages[source]["enc_hash"]
            return {client: dill.dumps(messages.keys()) for client in messages.keys()}

        if round_number == 4:
            out_ctxts = defaultdict(dict)

            for dest in messages.keys():
                for source in self.enc_hashes.keys():
                    signed = (
                        messages[source]["signed"]
                        if source in messages.keys()
                        else None
                    )
                    noises = (
                        self.noises[source] if source in self.noises.keys() else None
                    )
                    commitments = (
                        self.commitements[source]
                        if source in self.commitements.keys()
                        else None
                    )
                    out_ctxts[dest][source] = {
                        "enc_hash": self.enc_hashes[source],
                        "signed": signed,
                        "noise": noises,
                        "commitment": commitments,
                    }
            return out_ctxts

        if round_number == 5:
            decryptions = {c:messages[c]["partial_dec"] for c in messages.keys()}
            c2 = [messages[c]["c2"] for c in messages.keys()][0]
            self.sk_shares = [share for share in self.sk_shares for c in messages.keys() if c == share.idx]
            lag_coeffs = self.teg.ss.lagrange(self.sk_shares)
            hash_sum = self.teg.decrypt(decryptions, c2, lag_coeffs)
            self.total = np.sum(
                [self.masked_values[c] for c in self.masked_values.keys()], axis=0
            )
            return {
                client: {"agg_value": self.total, "hash_sum": hash_sum}
                for client in messages.keys()
            }

        if round_number == 6:
            assert all(messages.values())
            self.succeed(result=self.total)

    def distinct_points_compute(self, alphas):
        distinct_bases = tuple(alpha * self.P for alpha in alphas)
        return distinct_bases
