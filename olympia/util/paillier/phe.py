import random
from typing import List

from ..util import invert, powmod
from ..ss.util import get_two_safe_primes
from .phe_utils import (EncryptedNumberPaillier, PublicKeyPaillier,
                        SecretKeyPaillier, gen_coprime, l_function)

DEFAULT_KEY_SIZE = 2048


class PHE:
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def setup(self, lmbda: int = DEFAULT_KEY_SIZE):
        self.keysize: int = lmbda
        p, q = get_two_safe_primes(lmbda)
        pp = (p - 1) // 2
        qq = (q - 1) // 2
        p = 2 * pp + 1
        q = 2 * qq + 1
        n = p * q
        self.n = n
        self.phi_n = pp * qq
        g = n + 1
        beta = gen_coprime(n)

        secret_key = SecretKeyPaillier(self.phi_n * beta)
        theta = secret_key.s % n
        public_key = PublicKeyPaillier(n, g, theta)

        return public_key, secret_key

    def encrypt(self, pk: PublicKeyPaillier, m: int) -> EncryptedNumberPaillier:
        r = random.randint(1, pk.n - 1)
        c = (powmod(pk.g, m, pk.nsquared) * powmod(r, pk.n, pk.nsquared)) % pk.nsquared
        return EncryptedNumberPaillier(pk, c)

    def sum(self, ctxts: List[EncryptedNumberPaillier]) -> EncryptedNumberPaillier:
        assert len(ctxts) > 0, "No ciphertexts to sum"
        result = ctxts[0]
        for ctxt in ctxts[1:]:
            result = result * ctxt
        return result

    def decrypt(self, sk: SecretKeyPaillier, c: EncryptedNumberPaillier) -> int:
        return (
            l_function(powmod(c.c, sk.s, c.pk.nsquared), c.pk.n)
            * invert(c.pk.theta, c.pk.n)
        ) % c.pk.n
