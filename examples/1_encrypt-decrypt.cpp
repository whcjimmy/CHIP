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
In this example, we showed how to perform encryption and decryption of CKKS
scheme using HEaaN library.
*/
int main() {
    // You can use other parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    /*
    HEaaN::Message is a vector of complex numbers. The place where each element
    is located, is called 'slot'. 'log_slot' is the binary logarithm of the
    number of slots. 'log_slot' is necessary to construct HEaaN::Message class.
    The number of slots should be a power of two. 'full slots' is the maximum
    number of slots for given parameter.
    */
    const auto log_slots = getLogFullSlots(context);

    // Generate a new secret key
    HEaaN::SecretKey sk(context);

    /*
    You can also use the constuctors
    SecretKey(const Context &context, std::istream &stream) or
    SecretKey(const Context &context, const std::string &key_dir_path)
    if you have the saved secret key file.
    */

    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);

    std::cout << "Generate encryption key ... ";
    keygen.genEncryptionKey();
    std::cout << "done" << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    HEaaN::Message msg(log_slots);
    fillRandomComplex(msg);
    printMessage(msg);

    HEaaN::Ciphertext ctxt(context);
    std::cout << "Encrypt ... ";
    enc.encrypt(msg, pack, ctxt); // public key encryption
    std::cout << "done" << std::endl;

    HEaaN::Message dmsg;
    std::cout << "Decrypt ... ";
    dec.decrypt(ctxt, sk, dmsg);
    std::cout << "done" << std::endl;
    printMessage(dmsg);

    return 0;
}
