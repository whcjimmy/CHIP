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
In this example, we describe evaluating the average of numbers which is
encrypted into one ciphertext.
*/

int main() {
    // You can use other parameter instead of SS7.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::SS7;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);
    const size_t num_data = 30;

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate keys ... ";
    keygen.genEncryptionKey();
    keygen.genMultiplicationKey();
    // generate all rotation keys of index power of two
    keygen.genRotationKeyBundle();
    std::cout << "done" << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);
    HEaaN::HomEvaluator eval(context, pack);

    HEaaN::Message msg(log_slots);
    fillRandomReal(msg, num_data);

    std::cout << std::endl
              << "Data (" << num_data << " numbers) : " << std::endl;
    for (size_t i = 0; i < num_data - 1; ++i) {
        std::cout << msg[i].real() << ", ";
    }
    std::cout << msg[num_data - 1].real() << std::endl << std::endl;

    std::cout << "Encrypt ... ";
    HEaaN::Ciphertext ctxt(context);
    enc.encrypt(msg, pack, ctxt);
    std::cout << "done" << std::endl;

    std::cout << "Compute average ... ";
    {
        // smallest integer x such that num_data <= 2^x
        size_t rotation_required = std::ceil(std::log2(num_data));
        HEaaN::Ciphertext ctxt_rotated(context);
        for (size_t i = 0; i < rotation_required; ++i) {
            eval.leftRotate(ctxt, U64C(1) << i, ctxt_rotated);
            eval.add(ctxt, ctxt_rotated, ctxt);
        }

        eval.mult(ctxt, 1.0 / num_data, ctxt);
    }
    std::cout << "done" << std::endl;

    std::cout << "Decrypt ... ";
    HEaaN::Message dmsg;
    dec.decrypt(ctxt, sk, dmsg);
    std::cout << "done" << std::endl;

    std::cout << std::endl
              << "(decrypted) Average : " << dmsg[0].real() << std::endl;

    return 0;
}
