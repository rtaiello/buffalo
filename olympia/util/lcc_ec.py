import random
from collections import defaultdict

import numpy as np
from Crypto.PublicKey import ECC
from Crypto.PublicKey.ECC import EccPoint

from .util import invert, powmod


class LCCEllipticCurve:
    def __init__(self, curve: ECC._Curve):
        self.curve = curve
        self.point = EccPoint(self.curve.Gx, self.curve.Gy, self.curve.name)
        self.curve_order = int(curve.order)

    def _generate_point_ec(self):
        """Generates a random point on EC."""
        private_key = random.randint(1, self.curve_order)
        public_key_orig = private_key * self.point
        return public_key_orig

    def share(self, hash: ECC.EccPoint, threshold: int, nursers: int):
        "code inspiread from https://github.com/bbuyukates/LightVeriFL-fast/blob/main/ServerVerification/utils/EC.py"
        alpha_s = {i: i % self.curve_order for i in range(1, nursers + 1)}
        shares = defaultdict(EccPoint)
        coeffs = defaultdict(EccPoint)
        coeffs[0] = hash
        for i in range(1, threshold):
            coeffs[i] = self._generate_point_ec()

        for i in range(1, nursers + 1):
            shares[i] = coeffs[0]
            for j in range(1, threshold):
                shares[i] = shares[i] + coeffs[j] * powmod(
                    alpha_s[i], j, self.curve_order
                )
        return shares

    def pedersen_commitment(self, hash, noise):
        # Pedersen_g = scalar_mult(Pedersen_coeff[0], curve.g)
        # Pedersen_l = scalar_mult(Pedersen_coeff[1], curve.g)
        pedersen_coeff = np.zeros((2,), dtype=np.uint8)

        # Pedersen commitment coefficients
        pedersen_g = pedersen_coeff[0] * self.point
        pedersen_l = pedersen_coeff[1] * self.point
        pedersen_g = EccPoint(self.curve.Gx, self.curve.Gy, self.curve.name)
        pedersen_l = EccPoint(self.curve.Gx, self.curve.Gy, self.curve.name)

        tmp1 = noise + pedersen_g
        tmp2 = hash + pedersen_l
        commitment = tmp1 + tmp2
        return commitment, pedersen_g, pedersen_l

    def lagrange(self, threhsold: int, idx: list):
        "code inspiread from https://github.com/bbuyukates/LightVeriFL-fast/blob/main/ServerVerification/utils/EC.py"

        def PI(vals, p):
            accum = 1
            for v in vals:
                tmp = v % p  # np.mod(v, p)
                accum = (accum * tmp) % p  # np.mod(accum * tmp, p)
            return accum

        lag_coffs = defaultdict(EccPoint)
        alpha_s = {i: i % self.curve_order for i in idx[:threhsold]}
        for i in idx[:threhsold]:
            cur_alpha = alpha_s[i]

            den = PI(
                [cur_alpha - o for o in alpha_s if cur_alpha != o], self.curve_order
            )
            num = PI([0 - o for o in alpha_s if cur_alpha != o], self.curve_order)
            lag_coffs[i] = num * invert(den, self.curve_order)
        return lag_coffs

    def reconstruct(self, shares: dict, threshold: int, lag_coffs: list = None):
        "code inspiread from https://github.com/bbuyukates/LightVeriFL-fast/blob/main/ServerVerification/utils/EC.py"

        if lag_coffs is None:
            lag_coffs = self.lagrange(threshold, list(shares.keys()))
        idx = list(shares.keys())
        hash_recon = lag_coffs[idx[0]] * shares[idx[0]]
        hash_recon = EccPoint(hash_recon.x, hash_recon.y, self.curve.name)
        for i in idx[1:threshold]:
            tmp_elem = lag_coffs[i] * shares[i]
            tmp_elem = EccPoint(tmp_elem.x, tmp_elem.y, self.curve.name)
            hash_recon = hash_recon + tmp_elem
            hash_recon = EccPoint(hash_recon.x, hash_recon.y, self.curve.name)
        return hash_recon

    def sum_share(self, shares):
        accum = EccPoint(0, 0, curve=self.curve.name)
        for share in shares:
            temp = EccPoint(share.x, share.y, curve=self.curve.name)
            accum = accum + temp
            accum = EccPoint(accum.x, accum.y, curve=self.curve.name)
        return accum
