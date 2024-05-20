import random
from os import urandom as rng
from typing import List

from gmpy2 import is_prime, powmod

from .util import PField, SMALL_PRIMES, get_predefined_parameters
from .shamir_ss import Share
from collections import defaultdict


def is_power_of(x, base):
    p = 1
    while p < x:
        p = p * base
    return p == x


def sample_prime(bitsize):
    lower = 1 << (bitsize - 1)
    upper = 1 << (bitsize)
    while True:
        candidate = random.randrange(lower, upper)
        if is_prime(candidate):
            return candidate


def remove_factor(x, factor):
    while x % factor == 0:
        x //= factor
    return x


def prime_factor(x):
    factors = []
    for prime in SMALL_PRIMES:
        if prime > x:
            break
        if x % prime == 0:
            factors.append(prime)
            x = remove_factor(x, prime)
    assert x == 1  # fail if we were trying to factor a too large number
    return factors


def find_prime(min_bitsize, order_divisor):
    while True:
        k1 = sample_prime(min_bitsize)
        for k2 in range(128):
            q = k1 * k2 * order_divisor + 1
            if is_prime(q):
                order_prime_factors = [k1]
                order_prime_factors += prime_factor(k2)
                order_prime_factors += prime_factor(order_divisor)
                return q, order_prime_factors


def find_generator(q, order_prime_factors):
    order = q - 1
    for candidate in range(2, q):
        for factor in order_prime_factors:
            exponent = order // factor
            if pow(candidate, exponent, q) == 1:
                break
        else:
            return candidate


def find_prime_field(min_bitsize, order_divisor):
    q, order_prime_factors = find_prime(min_bitsize, order_divisor)
    g = find_generator(q, order_prime_factors)
    return q, g


def _find_parameters(min_bitsize, order2):
        order_divisor = order2
        q, g = find_prime_field(min_bitsize, order_divisor)
        assert is_prime(q)

        order = q - 1
        assert order % order2 == 0
        omega2 = pow(g, order // order2, q)

        return q, omega2
    
def generate_parameters(min_bitsize, order2):
    assert is_power_of(order2, 2), "order2 must be a power of 2"

    if min_bitsize == 2048 and order2 in [64, 128, 256, 512]:
        return get_predefined_parameters(order2)
    else:
        return _find_parameters(min_bitsize, order2)




class FTTField(PField):
    bits = None
    field_value = None

    def __init__(self, encoded_value):
        super().__init__(encoded_value, FTTField.field_value, FTTField.bits)

    # create a method to set the field value and bits
    @classmethod
    def set_field(cls, field_value, bits):
        cls.field_value = field_value
        cls.bits = bits


class FSSS(object):
    """Represents an implementation of the Fast Fourirer Transform Secret Sharing Scheme.

    Attributes:
        bitlength (int): The length of the bits for the secret sharing.
        Field (PField): The prime field used for the secret sharing.

    Methods:
        __init__(self, bitlength) -> None: Initializes a new SSS object.
        share(self, secret, threshold, nusers): Performs secret sharing.
        reconstruct(self, shares: int, threshold: int, lagcoefs: List[PField]): Reconstructs the secret.
        lagrange(self, shares): Calculates Lagrange coefficients for reconstructing the secret.

    """

    def __init__(self, nusers, min_bitsize_field) -> None:
        """Initializes a new SSS object.

        Args:
            bitlength (int): The length of the bits for the secret sharing.

        """
        super().__init__()
        self.nusers = nusers
        self.q, self.omega2 = generate_parameters(min_bitsize_field, nusers)

        FTTField.set_field(self.q, self.q.bit_length())
        self.field = FTTField
        self.bitlength = self.field.bits

    def _fft2_forward(self, aX, omega):
        if len(aX) == 1:
            return aX

        # split A(x) into B(x) and C(x) -- A(x) = B(x^2) + x C(x^2)
        bX = aX[0::2]
        cX = aX[1::2]

        # apply recursively
        omega_squared = powmod(omega, 2, self.q)
        B = self._fft2_forward(bX, omega_squared)
        C = self._fft2_forward(cX, omega_squared)

        # combine subresults
        A = [0] * len(aX)
        Nhalf = len(aX) >> 1
        point = 1
        for i in range(0, Nhalf):

            x = point
            A[i] = (B[i] + x * C[i]) % self.q
            A[i + Nhalf] = (B[i] - x * C[i]) % self.q

            point = (point * omega) % self.q

        return A

    def share(self, secret, threshold):
        """Performs secret sharing.

        Args:
            secret: The secret to be shared.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            nusers (int): The total number of shares to generate.

        Returns:
            List[Share]: A list of Share objects representing the shares.

        """
        small_coeffs = [secret] + [
            random.randrange(self.q) for _ in range(threshold - 1)
        ]
        large_coeffs = small_coeffs + [0] * (self.nusers - threshold)
        large_values = self._fft2_forward(large_coeffs, self.omega2)
        shares = defaultdict(dict)
        for idx, share in enumerate(large_values):
            value = powmod(self.omega2, idx, self.q)
            share_obj = Share(value, self.field(share))
            shares[idx+1] = share_obj
        return shares

    def reconstruct(self, shares: List[Share], threshold: int, lagcoefs: List[PField]):
        """Reconstructs the secret from the given shares using Lagrange interpolation.

        Args:
            shares (List[Share]): A list of Share objects representing the shares.
            threshold (int): The minimum number of shares required to reconstruct the secret.
            lagcoefs (List[PField]): A list containing the Lagrange coefficients.

        Returns:
            int: The reconstructed secret.

        Raises:
            AssertionError: If not enough shares are provided for reconstruction.

        """
        assert len(shares) >= threshold, "Not enough shares, cannot reconstruct!"
        raw_shares = []
        for x in shares:
            idx = self.field(x.idx)
            value = x.value
            if any(y[0] == idx for y in raw_shares):
                raise ValueError("Duplicate share")
            raw_shares.append((idx, value))
        k = len(shares)
        result = self.field(0)
        for j in range(k):
            x_j, y_j = raw_shares[j]
            result += y_j * lagcoefs[x_j]
        return result._value

    def lagrange(self, shares: List[Share]):
        """Calculates Lagrange coefficients for reconstructing the secret.

        Args:
            shares (List[Share]): A list of Share objects representing the shares.

        Returns:
            List[PField]: A list containing the Lagrange coefficients.

        Raises:
            ValueError: If duplicate shares are provided.

        """
        k = len(shares)
        indices: List[PField] = []
        for x in shares:
            idx = self.field(x.idx)
            if any(y == idx for y in indices):
                raise ValueError("Duplicate share")
            indices.append(idx)

        lag_coeffs = {}
        for j in range(k):
            x_j = indices[j]

            numerator = self.field(1)
            denominator = self.field(1)

            for m in range(k):
                x_m = indices[m]
                if m != j:
                    numerator *= x_m
                    denominator *= x_m - x_j
            lag_coeffs[x_j] = numerator * denominator.inverse()
        return lag_coeffs
