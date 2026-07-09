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

/*
In this example, we describe performing HEaaN on GPU. The following code
performs the same operations as the code in '6_bootstrap.cpp'.
*/
int main() {
    if (!HEaaN::CudaTools::isAvailable()) {
        std::cout << "Device is not available." << std::endl;
        return -1;
    }

    HEaaN::HEaaNTimer timer(true);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
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

    std::cout << "Load all public keys to GPU memory ..." << std::endl;
    timer.start("* ");
    pack.to(HEaaN::DeviceType::GPU);
    timer.end();
    std::cout << std::endl;

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
    std::cout << std::endl;

    std::cout << "Load boot constants data to GPU memory ..." << std::endl;
    timer.start("* ");
    eval.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
    timer.end();

    {
        HEaaN::Message msg(log_slots);
        fillRandomComplex(msg);

        std::cout << std::endl << "Input message : " << std::endl;
        printMessage(msg);
        std::cout << std::endl;

        /*
        Use 'to' function for loading object to GPU memory.
        For all HEaaN function, if input data is on GPU then output data is also
        on GPU.
        */
        msg.to(HEaaN::DeviceType::GPU);

        std::cout << "Encrypt ... ";
        HEaaN::Ciphertext ctxt(context);
        enc.encrypt(msg, pack, ctxt);
        std::cout << "done" << std::endl << std::endl;

        std::cout << "Input ciphertext - level " << ctxt.getLevel()
                  << std::endl;

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

        HEaaN::Ciphertext ctxt_boot(context);
        std::cout << "Bootstrapping ... " << std::endl;
        /*
        This is the bootstrapping for complex numbers.
        You can omit the third argument if encrypted messages are all real
        numbers. The bootstrapping for real numbers is faster.
        */
        timer.start("* ");
        eval.bootstrap(ctxt_rst, ctxt_boot, true);
        timer.end();
        std::cout << std::endl;

        std::cout << "Result ciphertext after bootstrapping - level "
                  << ctxt_boot.getLevel() << std::endl
                  << std::endl;

        /*
        Decrypt does not work well
        if ciphertext and key data are on different device memory.
        */
        sk.to(HEaaN::DeviceType::GPU);

        HEaaN::Message dmsg, dmsg_boot;
        std::cout << "Decrypt ... ";
        dec.decrypt(ctxt_rst, sk, dmsg);
        dec.decrypt(ctxt_boot, sk, dmsg_boot);
        std::cout << "done" << std::endl;

        /*
        Decrypted message data should be on CPU memory
        because below code is not working for GPU
        */
        dmsg.to(HEaaN::DeviceType::CPU);
        dmsg_boot.to(HEaaN::DeviceType::CPU);

        std::cout.precision(10);
        std::cout << std::endl << "Decrypted message : " << std::endl;
        printMessage(dmsg);
        std::cout << std::endl
                  << "Decrypted message (after bootstrapping) : " << std::endl;
        printMessage(dmsg_boot);
    }

    return 0;
}
