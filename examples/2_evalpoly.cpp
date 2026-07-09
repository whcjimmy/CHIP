////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "examples.hpp"

/*
In this example, we describe evaluating a polynomial function on encrypted input
data for equidistant points in the interval [0, 1].
*/

void print_polynomial(const double *coeff, const size_t degree) {
    auto print_coeff = [](double coeff) {
        if (coeff < 0.0L)
            std::cout << " - " << std::abs(coeff);
        else
            std::cout << " + " << coeff;
    };

    std::cout << coeff[degree] << "X^" << degree;
    for (size_t idx = degree - 1; idx > 1; --idx) {
        print_coeff(coeff[idx]);
        std::cout << "X^" << idx;
    }
    print_coeff(coeff[1]);
    std::cout << "X";
    print_coeff(coeff[0]);
}

int main() {
    // You can use other parameter instead of ST11.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::ST11;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);
    const auto num_slots = U64C(1) << log_slots;

    // This is the approximation polynomial of degree 7
    // for sigmoid function 1 / (1 + exp(-x)).
    constexpr const HEaaN::Real POLY_COEFF[] = {
        0.5, 0.220557256,    0.0, -8.55553085e-03,
        0.0, 1.74370698e-04, 0.0, -1.24789854e-06};
    constexpr const size_t POLY_DEGREE = 7;

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... ";
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl;

    std::cout << "Generate multiplication key ... ";
    keygen.genMultiplicationKey();
    std::cout << "done" << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);
    HEaaN::HomEvaluator eval(context, pack);

    HEaaN::Message msg(log_slots);

    std::cout << std::endl << "Number of slots = " << num_slots << std::endl;

    for (size_t i = 0; i < num_slots; ++i) {
        msg[i].real((double)i / num_slots);
        msg[i].imag(0.0);
    }

    std::cout << std::fixed;
    std::cout.precision(7);
    std::cout << std::endl << "Input vector : " << std::endl;
    printMessage(msg, false); // print the numbers with only real part
    std::cout << std::endl;

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt);
    std::cout << "done" << std::endl;

    std::cout << "Multiplicative depth : " << ctxt.getLevel() << std::endl
              << std::endl;

    HEaaN::Ciphertext ctxt_rst(context);

    std::cout << "Evaluating polynomial " << std::endl;
    print_polynomial(POLY_COEFF, POLY_DEGREE);
    std::cout << std::endl << " ... ";

    {
        /*
        Evaluate polynomial using Horner's method
        c_0 + X * (c_1 + X * (c_2 + ... X * (c_{n-1} + X * c_n) ... ))
        This algorithm spends as many levels as the degree of polynomial.
        It is not optimized wih respect to level.
        There exists an algorithm which spends only log_2(degree + 1) levels.
        */
        size_t idx = POLY_DEGREE;
        do {
            idx--;
            if (idx == POLY_DEGREE - 1) {
                eval.mult(ctxt, POLY_COEFF[POLY_DEGREE], ctxt_rst);
            } else {
                eval.mult(ctxt_rst, ctxt, ctxt_rst);
            }
            eval.add(ctxt_rst, POLY_COEFF[idx], ctxt_rst);
        } while (idx != 0);
    }
    std::cout << "done" << std::endl;

    std::cout << "Remaining multiplicative depth : " << ctxt_rst.getLevel()
              << std::endl
              << std::endl;

    HEaaN::Message dmsg;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt_rst, sk, dmsg);
    std::cout << "done" << std::endl;

    std::cout << std::endl << "Result vector : " << std::endl;
    printMessage(dmsg, false);

    return 0;
}
