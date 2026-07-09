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
In this example, we describe saving a ciphertext to file and loading the
ciphertext from the file.
*/
int main() {
    // You can use other parameter instead of ST11.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::ST11;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);

    // Save a ciphertext to file
    {
        std::cout << "Generate secret key and save it to file ... ";
        HEaaN::SecretKey sk(context);
        sk.save("secretkey.bin");
        std::cout << "done" << std::endl;

        HEaaN::KeyPack pack(context);
        HEaaN::KeyGenerator keygen(context, sk, pack);

        std::cout << "Generate encryption key ... ";
        keygen.genEncryptionKey();
        std::cout << "done" << std::endl;

        HEaaN::Encryptor enc(context);

        HEaaN::Message msg(log_slots);
        fillRandomComplex(msg);

        std::cout << std::endl << "Input message : " << std::endl;
        printMessage(msg);
        std::cout << std::endl;

        HEaaN::Ciphertext ctxt(context);
        std::cout << "Encrypt ... ";
        enc.encrypt(msg, pack, ctxt); // public key encryption
        std::cout << "done" << std::endl;

        std::cout << "Save the ciphertext to file ... ";
        ctxt.save("ctxt0.bin");
        std::cout << "done" << std::endl << std::endl;
    }

    // Load the ciphertext from file
    {
        /*
        To decrypt loaded ciphertext, we need the same secret key as above.
        */
        std::cout << "Load secret key from file ... ";
        HEaaN::SecretKey sk(context, "./secretkey.bin");
        std::cout << "done" << std::endl;

        HEaaN::Decryptor dec(context);

        HEaaN::Ciphertext ctxt(context);

        std::cout << "Load the ciphertext from file  ... ";
        ctxt.load("./ctxt0.bin");
        std::cout << "done" << std::endl << std::endl;

        HEaaN::Message dmsg;
        std::cout << "Decrypt ... ";
        dec.decrypt(ctxt, sk, dmsg);
        std::cout << "done" << std::endl << std::endl;

        std::cout << "Decrypted message : " << std::endl;
        printMessage(dmsg);
    }

    return 0;
}
