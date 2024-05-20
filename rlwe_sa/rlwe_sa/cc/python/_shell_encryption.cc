#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "rlwe_sa/cc/shell_encryption_api.h"  // Include the header file where your class is defined
#include "rlwe_sa/cc/shell_encryption/montgomery.h"
#include "absl/numeric/int128.h"


namespace rlwe_sa {

namespace py = pybind11;
using ModularInt = rlwe::MontgomeryInt<uint64_t>;


PYBIND11_MODULE(_shell_encryption, m) {
    // Bind the class
    py::class_<RlweSecAgg<ModularInt>>(m, "RlweSecAgg")
        // Constructor int,size_t and optional string
        .def(py::init<int, size_t>(), "Constructor with two arguments")
        .def(py::init<int, size_t, std::string>(), "Constructor with three arguments")
        // Return get_seed as bytes
        .def("get_seed", [](RlweSecAgg<ModularInt> &instance) {
            std::string seed = instance.GetSeed();  // Get the seed as a std::string
            return py::bytes(seed);  // Return the data as py::bytes without transcoding
        })
        .def("sample_key", &RlweSecAgg<ModularInt>::SampleKey)  // Member function
        .def("create_key", &RlweSecAgg<ModularInt>::CreateKey)  // Member function
        .def("encrypt", &RlweSecAgg<ModularInt>::Encrypt)  // Member function
        .def("decrypt", &RlweSecAgg<ModularInt>::Decrypt)  // Member function
        .def("aggregate", &RlweSecAgg<ModularInt>::Aggregate)  // Member function
        .def("sum_keys", &RlweSecAgg<ModularInt>::SumKeys)  // Member function
        .def_static("sample_plaintext", &RlweSecAgg<ModularInt>::SamplePlaintext)  // Static function
        .def_static("convert_key", &RlweSecAgg<ModularInt>::ConvertKey);  // Static function
    py::class_<rlwe::SymmetricRlweKey<ModularInt>>(m, "SymmetricRlweKey");
    py::class_<rlwe::SymmetricRlweCiphertext<ModularInt>>(m, "SymmetricRlweCiphertext")
    // Add this two method Len and LogModulus
        .def("len", &rlwe::SymmetricRlweCiphertext<ModularInt>::Len)
        .def("log_modulus", &rlwe::SymmetricRlweCiphertext<ModularInt>::LogModulus)
        .def("num_coeffs", &rlwe::SymmetricRlweCiphertext<ModularInt>::NumCoeffs);
}
}