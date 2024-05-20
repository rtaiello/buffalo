import subprocess

from gmpy2 import is_prime, mpz


def generate_safe_prime(bits):
    openssl_command = f"openssl prime -generate -bits {bits} -safe"

    try:
        output = subprocess.check_output(openssl_command, shell=True, text=True)
        safe_prime_str = output.strip()
        safe_prime_int = int(safe_prime_str)
        return safe_prime_int
    except subprocess.CalledProcessError:
        print("Error: Unable to generate a safe prime number.")
        return None


def main():
    bits = 1024  # Adjust the number of bits for the desired safe prime size
    safe_prime = generate_safe_prime(bits)
    assert is_prime(mpz(safe_prime))
    if safe_prime:
        print(f"Generated safe prime with {bits} bits:")
        print(safe_prime)


if __name__ == "__main__":
    main()
