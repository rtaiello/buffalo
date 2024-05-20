from math import factorial
from typing import List, Optional

from ...ss.integer_ss import ISSS, IShare
from ...util import invert, powmod
from ..phe import PHE
from ..phe_utils import (EncryptedNumberPaillier, PublicKeyPaillier,
                         SecretKeyPaillier, l_function)


class TPHE(PHE):
    def __init__(self, threshold: int, nusers: int, sigma: int = 128) -> None:
        super().__init__()
        self.threshold = threshold
        self.nusers = nusers
        self.delta = factorial(self.nusers)
        self.sigma = sigma
        self.iss: Optional[ISSS] = None

    def sk_share(self, sk: SecretKeyPaillier) -> List[IShare]:
        assert self.n, "Setup not called"
        self.iss = ISSS(self.keysize, self.sigma)
        return self.iss.share(sk.s, self.threshold, self.nusers)

    def partial_decrypt(
        self, sk_share: IShare, cipher_text: EncryptedNumberPaillier
    ) -> int:
        partial_decryption = powmod(
            cipher_text.c,
            sk_share.value,
            cipher_text.pk.nsquared,
        )
        return IShare(sk_share.idx, partial_decryption)

    def decrypt(
        self,
        pk: PublicKeyPaillier,
        partial_decryptions: List[IShare],
        lag_coeffs=None,
    ) -> int:
        if not lag_coeffs:
            lag_coeffs = self.iss.lagrange(partial_decryptions, self.delta)
        product = 1
        for lag_coeff, partial_decryption in zip(
            lag_coeffs.values(), partial_decryptions
        ):
            product = (
                product
                * powmod(partial_decryption.value, lag_coeff, pk.nsquared)
                % pk.nsquared
            )
        inv_temp = invert(self.delta**2 * pk.theta % pk.n, pk.n)
        m = l_function(product, pk.n) * inv_temp % pk.n

        if m > (pk.n // 2):
            m -= pk.n
        return m
