import random
from math import gcd


class PublicKeyPaillier:
    def __init__(self, n, g, theta=None):
        self.n = n
        self.nsquared = n**2
        self.g = g
        self.theta = theta

    def __eq__(self, __value: object) -> bool:
        return self.n == __value.n and self.g == __value.g

    def set_theta(self, theta):
        self.theta = theta


class SecretKeyPaillier:
    def __init__(self, s):
        self.s = s


class EncryptedNumberPaillier:
    def __init__(self, pk, c):
        self.pk = pk
        self.c = c

    def __mul__(self, other):
        if isinstance(other, EncryptedNumberPaillier):
            return EncryptedNumberPaillier(
                self.pk, (self.c * other.c) % self.pk.nsquared
            )
        else:
            return EncryptedNumberPaillier(self.pk, (self.c * other) % self.pk.nsquared)

    def get_real_size(self):
        return self.c.bit_length()


def l_function(x, n):
    return (x - 1) // n


def gen_coprime(n):
    random.seed(0)
    while True:
        ret = random.randint(0, n - 1)
        if gcd(ret, n) == 1:
            return ret
