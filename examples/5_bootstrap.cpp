////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright (C) 2021-2022 Crypto Lab Inc.                                    //
//                                                                            //
// - This file is part of HEaaN homomorphic encryption library.               //
// - HEaaN cannot be copied and/or distributed without the express permission //
//  of Crypto Lab Inc.                                                        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include "omp.h"

/*
In this example, we describe bootstrapping a ciphertext on GPU memory.
*/
int main() {
    omp_set_num_threads(64);
    HEaaN::HEaaNTimer timer(false);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    if (!HEaaN::isBootstrappableParameter(context)) {
        std::cout << "Bootstrap is not available for parameter "
                  << presetNamer(preset) << std::endl;
        return -1;
    }

    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl << std::endl;

    /*
    You should perform makeBootstrappble function
    before generating evaluation keys and constucting HomEvaluator class.
    */
    HEaaN::makeBootstrappable(context);

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
                 "bootstrapping) ..."
              << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    timer.end();

    {
        /*
        HEaaN::Message msg(log_slots);
        fillRandomComplex(msg);

        std::cout << std::endl << "Input message : " << std::endl;
        printMessage(msg);
        std::cout << std::endl;

        std::cout << "Encrypt ... ";
        HEaaN::Ciphertext ctxt(context);
        enc.encrypt(msg, pack, ctxt);
        std::cout << "done" << std::endl << std::endl;
        */

        int length = 1024;
        std::vector<HEaaN::Message> input;

        std::cout << "create input\n";
        for(int i = 0; i < length; i++) {
            input.push_back(HEaaN::Message(log_slots));
            fillRandomComplex(input[i]);
        }

        std::vector<HEaaN::Ciphertext> input_cipher, input_cipher_bs;

        for(int i = 0; i < length; i++) {
            input_cipher.push_back(HEaaN::Ciphertext(context));
            input_cipher_bs.push_back(HEaaN::Ciphertext(context));
        }
#pragma omp parallel for
        for(int i = 0; i < length; i++) {
            enc.encrypt(input[i], pack, input_cipher[i]);
            eval.levelDown(input_cipher[i], 3, input_cipher[i]);
        }

        std::cout << "Input ciphertext - level " << input_cipher[0].getLevel()
                  << std::endl;

        total_time = 0.0;
        timer.start("* ");
        std::cout << "Bootstrapping ... " << std::endl;
#pragma omp parallel for 
        for(int i = 0; i < length; i++) {
            eval.bootstrap(input_cipher[i], input_cipher_bs[i]);
        }
        timer.end();
        std::cout << total_time << std::endl;

        /*
        // consume level with computation
        std::cout << std::endl
                  << "Computing the polynomial 1.2*X^4 - X + 1 ... ";
        HEaaN::Ciphertext ctxt2(context), ctxt4(context), ctxt_rst(context);
        eval.mult(ctxt, ctxt, ctxt2);
        eval.mult(ctxt2, ctxt2, ctxt4);
        eval.mult(ctxt4, 1.2, ctxt_rst);
        eval.sub(ctxt_rst, ctxt, ctxt_rst);
        eval.add(ctxt_rst, 1, ctxt_rst);
        std::cout << "done" << std::endl;

        std::cout << "Result ciphertext - level " << ctxt_rst.getLevel()
                  << std::endl
                  << std::endl;
        */

        /*
        HEaaN::Ciphertext ctxt_boot(context);
        std::cout << "Bootstrapping ... " << std::endl;
        timer.start("* ");
        eval.bootstrap(ctxt_rst, ctxt_boot, true);
        timer.end();
        std::cout << std::endl;
        */

        std::cout << "Result ciphertext after bootstrapping - level "
                  << input_cipher_bs[0].getLevel() << std::endl
                  << std::endl;

        for(int i = 0; i < 10; i++) {
            std::cout << i << std::endl;

            HEaaN::Message dmsg, dmsg_boot;
            std::cout << "Decrypt ... ";
            dec.decrypt(input_cipher[i], sk, dmsg);
            dec.decrypt(input_cipher_bs[i], sk, dmsg_boot);
            std::cout << "done" << std::endl;

            std::cout.precision(10);
            std::cout << std::endl << "Decrypted message : " << std::endl;
            printMessage(dmsg);
            std::cout << std::endl
                << "Decrypted message (after bootstrapping) : " << std::endl;
            printMessage(dmsg_boot);
        }

    }

    return 0;
}
