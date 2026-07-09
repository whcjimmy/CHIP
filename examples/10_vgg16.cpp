#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include "conv.hpp"
#include <HEaaN/Bootstrapper.hpp>
#include <HEaaN/Ciphertext.hpp>
#include <HEaaN/EnDecoder.hpp>
#include <HEaaN/Message.hpp>
#include <HEaaN/SecretKey.hpp>
#include <HEaaN/device/CudaTools.hpp>
#include <HEaaN/device/Device.hpp>
#include <cmath>
#include <highfive/H5Easy.hpp>
// #include "/usr/local/cuda/include/cuda_runtime.h"

extern double total_time;
double vm, vm2, rss, rss2;

void vgg_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
               std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
               std::vector<char> &vgg_meanpool,
               std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b1,
               std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b2,
               std::vector<double> &bias_b1,
               std::vector<double> &bias_b2,
               std::vector<double> &ss,
               std::vector<std::vector<double>> &u,
               std::vector<std::vector<double>> &v,
               bool bitm,
               HEaaN::Context &context, HEaaN::HomEvaluator &eval);

void vgg_cipher(std::vector<HEaaN::Ciphertext> &vec_cipher_1,
                std::vector<HEaaN::Ciphertext> &vec_cipher_2,
                std::vector<char> &vgg_meanpool,
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1,
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2,
                std::vector<double> bias_b1,
                std::vector<double> bias_b2,
                std::vector<double> ss,
                std::vector<std::vector<double>> u,
                std::vector<std::vector<double>> v,
                int input_size, int rp_pack_size, int chnl_batch_size, bool bitm, bool is_gpu,
                HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc,
                HEaaN::Decryptor &dec, HEaaN::SecretKey &sk, HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder);

int main(int argc, char *argv[]) {
    omp_set_num_threads(40);
    int is_gpu = false;
    std::string type = "A";
    int batch_point = atoi(argv[1]);
    std::cout << "Batch starts from No." << batch_point << std::endl;

    std::cout << "========\n";
    if(is_gpu) {
        std::cout << "GPU MODE\n";
    }else{
        std::cout << "CPU MODE\n";
    }
    std::cout << "========\n";

    HEaaN::HEaaNTimer timer(true);
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
    const auto num_slots = UINT64_C(1) << log_slots;

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);
    HEaaN::EnDecoder edcoder(context);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    keygen.genRotKeysForBootstrap(log_slots);
    std::cout << "done" << std::endl << std::endl;

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
    HEaaN::Bootstrapper boot(eval, log_slots);
    // HEaaN::makeBootstrappable(context);
    timer.end();
    timer.print();

    {
        std::cout << "==========\n";
        std::cout << "READ INPUT\n";
        std::cout << "==========\n";
        std::vector<std::vector<std::vector<std::vector<double>>>> input_double(10000, 
                                  std::vector<std::vector<std::vector<double>>>(3, 
                                               std::vector<std::vector<double>>(32, 
                                                            std::vector<double>(32, 0.0))));
        std::vector<double> label(10000);
        std::vector<std::vector<std::vector<HEaaN::Message>>> input(3, 
                           std::vector<std::vector<HEaaN::Message>>(34, 
                                        std::vector<HEaaN::Message>(34, 
                                                                    HEaaN::Message(log_slots))));
        std::vector<std::vector<std::vector<HEaaN::Message>>> output(64, 
                            std::vector<std::vector<HEaaN::Message>>(32, 
                                         std::vector<HEaaN::Message>(32, 
                                                                     HEaaN::Message(log_slots))));
        // Open the HDF5 INPUT FILE
        // Read input and label
        HighFive::File file_2("../../model/input_c100.h5", HighFive::File::ReadOnly);
        file_2.getDataSet("input").read(input_double);
        file_2.getDataSet("label").read(label);
        for(int b = 0; b < 64; b++) {
            for(int ic = 0; ic < 3; ic++) {
                for(int i = 0; i < 34; i++) {
                    for(int j = 0; j < 34; j++) {
                        if(i == 0 || j == 0 || i == 33 || j == 33) {
                            fillZero(input[ic][i][j]);
                        }else{
                            input[ic][i][j][b].real(input_double[batch_point + b][ic][i - 1][j - 1]);
                            input[ic][i][j][b].imag(0);
                        }
                    }
                }
            }
        }

        // ENCRYPT INPUT
        process_mem_usage(vm, rss);
        std::vector<HEaaN::Ciphertext> input_cipher(34 * 34, HEaaN::Ciphertext(context));
        std::vector<HEaaN::Ciphertext> output_cipher(32 * 32 , HEaaN::Ciphertext(context));
#pragma omp parallel for collapse(2)
        for(int i = 0; i < 34; i++) {
            for(int j = 0; j < 34; j++) {
                HEaaN::Message msg(log_slots);
                fillZero(msg);
                for(int b = 0; b < 64; b++) {
                    for(int oc = 0; oc < 64; oc++) {
                        for(int ic = 0; ic < 4; ic++) {
                            int pos = (type == "A") ?  
                                pos = b * 512 + ic * 64 + oc : 
                                pos = b * 512 + oc * 4 + ic;
                            if(ic < 3) {
                               msg[pos] = input[ic][i][j][b];
                            }else{
                                msg[pos].real(0);
                            }
                        }
                    }
                }
                int enc_pos = i * 34 + j;
                enc.encrypt(msg, pack, input_cipher[enc_pos]);
                eval.levelDown(input_cipher[enc_pos], 11, input_cipher[enc_pos]);
            }
        }
        process_mem_usage(vm2, rss2);
        std::cout << "INPUT and OUTPUT CIPHER RAM USAGE " << (vm2 - vm)/(1024 * 1024) << " GB" << std::endl;
        std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;
        std::cout << "input_cipher level " << input_cipher[0].getLevel() << std::endl;

        std::cout << "==========\n";
        std::cout << "READ MODEL\n";
        std::cout << "==========\n";
        std::vector<std::vector<std::vector<std::vector<double>>>> kernel(64, 
                            std::vector<std::vector<std::vector<double>>>(3, 
                                         std::vector<std::vector<double>>(3, 
                                                      std::vector<double>(3, 0.0))));
        std::vector<double> bias(64, 0.0);
        std::vector<double> act(3);
        double scale, shift;
        std::vector<double>u(2);
        std::vector<double>v(2);

        // Open the HDF5 MODEL file
        HighFive::File file("../../model/c100_vgg16_7055.h5", HighFive::File::ReadOnly);
        // Read dataset
        file.getDataSet("module.layer1.0.weight").read(kernel);
        file.getDataSet("module.layer1.0.bias").read(bias);
        file.getDataSet("module.layer1.1.scale").read(scale);
        file.getDataSet("module.layer1.1.shift").read(shift);
        file.getDataSet("module.layer1.1.mean").read(u);
        file.getDataSet("module.layer1.1.var").read(v);

        std::cout << "================\n";
        std::cout << "PLAINTEXT DOMAIN\n";
        std::cout << "================\n";
        // First Conv
        conv_plain(input, output, kernel, bias, 1, context, eval);
        cal_act(act, u, v, scale, shift);
        act_plain(output, act, context, eval);

        resize_3d(input, 32, 32, 64, context);
        copy_pt(input, output);
        
        std::cout << "=================\n";
        std::cout << "ENCRYPTION DOMAIN\n";
        std::cout << "=================\n";
        // First Conv
        cal_act(act, u, v, scale, shift);
        update_model(kernel, bias, act);

        FCA(input_cipher, output_cipher, kernel, bias, 34, 34, 4, true, type, is_gpu, context, pack, enc, eval, edcoder);

        process_mem_usage(vm2, rss2);
        std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;

        act_cipher(output_cipher, act, 32, 32, context, edcoder, eval);
        copy_ct(input_cipher, output_cipher, 32, 32);
        print_ct(input_cipher, 32, 32, 3, 64, type, dec, sk);

        std::cout << "input_cipher level " << input_cipher[0].getLevel() << std::endl;

        /*
        int kernel_oc_n = kernel_oc; 
        SPLIT(input_cipher, dca_output_row, dca_output_col, kernel_ic, kernel_oc, kernel_oc_n, type, is_gpu, context, pack, enc, eval, edcoder);
        IR(input_cipher, dca_output_row, dca_output_col, kernel_ic, kernel_oc, kernel_oc_n, type, is_gpu, context, pack, enc, eval, edcoder);

        std::cout << "IMAGE REARRANGEMENT\n";
        // HEaaN::Message dmsg;
        for(int i = 0; i < dca_output_row; i++) {
            for(int j = 0; j < dca_output_col; j++) {
                int output_pos = i * dca_output_col + j;
                dec.decrypt(input_cipher[output_pos], sk, dmsg);
                for(int ii = 0; ii < kernel_oc * kernel_oc_n + 10; ii++) {
                    std::cout << dmsg[ii].real() << " " ;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        std::cout << "input_cipher level " << input_cipher[0].getLevel() << std::endl;
        */

        std::vector<std::vector<int>> vgg_struct = {{64, 128}, {128, 256}, {256, 256}, {512, 512}, {512, 512}, {512, 512}};
        std::vector<std::vector<char>> vgg_meanpool = {{'M', '0'}, {'M', '0'}, {'0', 'M'}, {'0', '0'}, {'M', '0'}, {'0', 'M'}};
        std::vector<std::vector<std::string>> vgg_name = {{"1.2", "1.3", "2.0", "2.1"}, 
                                                          {"2.2", "2.3", "3.0", "3.1"}, 
                                                          {"3.2", "3.3", "3.4", "3.5"},
                                                          {"4.0", "4.1", "4.2", "4.3"},
                                                          {"4.4", "4.5", "5.0", "5.1"},
                                                          {"5.2", "5.3", "5.4", "5.5"}};
        // VGG
        int vgg_ic_1 = 64;
        int vgg_oc_1 = 64;
        int vgg_ic_2 = 64;
        int vgg_oc_2 = 64;
        int vgg_input_size = 32;
        int vgg_output_size = 32;
        
        std::vector<HEaaN::Ciphertext> input_cipher_2(vgg_input_size * vgg_input_size, HEaaN::Ciphertext(context));
        std::vector<HEaaN::Ciphertext> output_cipher_2(vgg_output_size * vgg_output_size, HEaaN::Ciphertext(context));

        if(is_gpu) {
            std::cout << "Load boot constants data to GPU memory ..." << std::endl;
            boot.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
            // eval.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
        }

        HEaaN::Message zero(log_slots);
        fillZero(zero);

        for(int layer = 0; layer < vgg_struct.size(); layer++) {
                std::cout << "=====\n";
                std::cout << "=====\n";
                std::cout << "VGG " << layer << "\n";
                std::cout << "=====\n";
                std::cout << "=====\n";

                vgg_input_size = vgg_output_size;
                if('M' == vgg_meanpool[layer][0]) {
                    vgg_output_size = vgg_input_size / 2;
                }else if ('M' == vgg_meanpool[layer][1]){
                    vgg_output_size = vgg_input_size;
                }else{
                    // no meanpooling
                    vgg_output_size = vgg_input_size;
                }

                vgg_ic_1 = vgg_oc_2;
                vgg_oc_1 = vgg_struct[layer][0];
                vgg_ic_2 = vgg_struct[layer][0];
                vgg_oc_2 = vgg_struct[layer][1];

                int rp_pack_size = vgg_struct[layer][1];
                int batch_size;
                int chnl_batch_size; // the num of batch to process the whole num of channels
                int split_chnl_size; // the num of input chanels each batch can process
                if(vgg_struct[layer][0] * vgg_struct[layer][1] <= 32768) {
                    batch_size = 64/(32768/(vgg_struct[layer][0] * vgg_struct[layer][1]));
                    chnl_batch_size = 1;
                    split_chnl_size = vgg_ic_1;
                }else{
                    batch_size = 64;
                    chnl_batch_size = vgg_struct[layer][0] * vgg_struct[layer][1] / 32768;
                    split_chnl_size = 32768 / rp_pack_size;
                }

                bool bitm = false; // always false in VGG16

                std::cout << vgg_input_size << " " << vgg_output_size << std::endl;
                std::cout << vgg_ic_1 << " " << vgg_oc_1 << " " << vgg_ic_2 << " " << vgg_oc_2 << std::endl;
                std::cout << rp_pack_size << " " << batch_size << " " << chnl_batch_size << std::endl;
                std::cout << bitm << std::endl;

                // prepare weight and bias
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1(vgg_oc_1, 
                        std::vector<std::vector<std::vector<double>>>(vgg_ic_1, 
                            std::vector<std::vector<double>>(3, 
                                std::vector<double>(3, 0.0))));
                std::vector<double> bias_b1(vgg_oc_1, 0.0);
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2(vgg_oc_2, 
                        std::vector<std::vector<std::vector<double>>>(vgg_ic_2, 
                            std::vector<std::vector<double>>(3, 
                                std::vector<double>(3, 0.0))));
                std::vector<double> bias_b2(vgg_oc_2, 0.0);

                std::vector<double> ss(4);
                std::vector<std::vector<double>>u(2, std::vector<double>(2));
                std::vector<std::vector<double>>v(2, std::vector<double>(2));
                std::vector<double> act(3);

                // Read from pre-trained model
                std::string str = "module.layer" + vgg_name[layer][0] + ".";
                file.getDataSet(str + "weight").read(kernel_b1);
                file.getDataSet(str + "bias").read(bias_b1);
                str = "module.layer" + vgg_name[layer][2] + ".";
                file.getDataSet(str + "weight").read(kernel_b2);
                file.getDataSet(str + "bias").read(bias_b2);
                str = "module.layer" + vgg_name[layer][1] + ".";
                file.getDataSet(str + "scale").read(ss[0]);
                file.getDataSet(str + "shift").read(ss[1]);
                file.getDataSet(str + "mean").read(u[0]);
                file.getDataSet(str + "var").read(v[0]);
                str = "module.layer" + vgg_name[layer][3] + ".";
                file.getDataSet(str + "scale").read(ss[2]);
                file.getDataSet(str + "shift").read(ss[3]);
                file.getDataSet(str + "mean").read(u[1]);
                file.getDataSet(str + "var").read(v[1]);

                std::cout << "================\n";
                std::cout << "PLAINTEXT DOMAIN\n";
                std::cout << "================\n";

                resize_3d(output, vgg_input_size, vgg_input_size, vgg_oc_1, context);

                vgg_plain(input, output, vgg_meanpool[layer], kernel_b1, kernel_b2, bias_b1, bias_b2, ss, u, v, bitm, context, eval);

                if ('M' == vgg_meanpool[layer][1]){
                    resize_3d(input, vgg_output_size/2, vgg_output_size/2, vgg_oc_2, context);
                }else{
                    resize_3d(input, vgg_output_size, vgg_output_size, vgg_oc_2, context);
                }
                copy_pt(input, output);

                std::cout << "=================\n";
                std::cout << "ENCRYPTION DOMAIN\n";
                std::cout << "=================\n";

                output_cipher.resize(vgg_output_size * vgg_output_size, HEaaN::Ciphertext(context));
                output_cipher.shrink_to_fit();
                output_cipher_2.resize(vgg_input_size * vgg_input_size, HEaaN::Ciphertext(context));
                output_cipher_2.shrink_to_fit();
#pragma omp parallel for
                for(int i = 0; i < vgg_output_size * vgg_output_size; i++) {
                    enc.encrypt(zero, pack, output_cipher[i]);
                    eval.levelDown(output_cipher[i], 3, output_cipher[i]);
                }

                for(int b = 0; b < batch_size; b++) {
                // for(int b = 0; b < 1; b++) {
                    std::cout << "============= " << std::endl;
                    std::cout << "BATCH SIZE " << b << std::endl;
                    std::cout << "============= " << std::endl;
                    // SPLIT (1->many)
                    SPLIT(input_cipher, input_cipher_2, vgg_input_size, vgg_input_size, vgg_oc_1, rp_pack_size, b * 512, "A", is_gpu, context, pack, enc, eval, edcoder);

                    IRA2B(input_cipher_2, vgg_input_size, vgg_input_size, rp_pack_size, split_chnl_size, is_gpu, context, pack, enc, eval, edcoder);
                    print_ct(input_cipher_2, vgg_input_size, vgg_input_size, rp_pack_size, vgg_oc_1, "A", dec, sk);

                    vgg_cipher(input_cipher_2, output_cipher_2, vgg_meanpool[layer], kernel_b1, kernel_b2, bias_b1, bias_b2,
                            ss, u, v, vgg_input_size, rp_pack_size, chnl_batch_size, bitm, is_gpu, context, pack, enc, dec, sk, eval, edcoder);

                    // MERGE (many->1)
                    int rp_pack_size_2 = rp_pack_size;
                    if(layer == 0){
                        rp_pack_size_2 = rp_pack_size / 2; // hotfix
                    }
                    MERGE(output_cipher_2, output_cipher, vgg_output_size, vgg_output_size, vgg_oc_2, rp_pack_size_2, b * 512, "A", is_gpu, context, pack, enc, eval, edcoder);
                }

                input_cipher.resize(vgg_output_size * vgg_output_size, HEaaN::Ciphertext(context));
                input_cipher.shrink_to_fit();
                input_cipher_2.resize(vgg_output_size * vgg_output_size, HEaaN::Ciphertext(context));
                input_cipher_2.shrink_to_fit();
                copy_ct(input_cipher, output_cipher, vgg_output_size, vgg_output_size);

                std::cout << "BOOTSTRAPPING\n";
                HEaaN::HEaaNTimer timer(true);
                timer.start("BOOTSTRAPPING ");
#pragma omp parallel for if(!is_gpu)
                for(int i = 0; i < vgg_output_size * vgg_output_size; i++) {
                    // eval.bootstrap(input_cipher[i], input_cipher[i]);
                    boot.bootstrap(input_cipher[i], input_cipher[i]);
                    if(layer < vgg_struct.size() - 2) {
                        eval.levelDown(input_cipher[i], 10, input_cipher[i]);
                    }else{
                        eval.levelDown(input_cipher[i], 11, input_cipher[i]);
                    }
                }
                std::cout << "LLLLLLLL " << input_cipher[0].getLevel() << std::endl;
                timer.end();
                timer.print();

                if(!bitm) {
                    cal_act(act, u[1], v[1], ss[2], ss[3]);
                    if('M' == vgg_meanpool[layer][0]) {
                        act[1] = act[1] / sqrt(act[0]);
                    }else if ('M' == vgg_meanpool[layer][1]) {
                        act[1] = act[1] / (sqrt(act[0]) * 2);
                        act[2] = act[2] / 4;
                    }else{
                        act[1] = act[1] / sqrt(act[0]);
                    }
                    act_cipher(input_cipher, act, vgg_output_size, vgg_output_size, context, edcoder, eval);

                    if ('M' == vgg_meanpool[layer][1]){
                        meanpool_cipher(input_cipher, output_cipher, vgg_output_size, vgg_output_size, context, pack, enc, eval);
                        vgg_output_size = vgg_output_size / 2;
                        copy_ct(input_cipher, output_cipher, vgg_output_size, vgg_output_size);
                    }
                }
                print_ct(input_cipher, vgg_output_size, vgg_output_size, rp_pack_size, vgg_oc_2, "A", dec, sk);
        }

        /*
        // Read Act
        file.getDataSet("module.classifier.1.scale").read(scale);
        file.getDataSet("module.classifier.1.shift").read(shift);
        file.getDataSet("module.classifier.1.mean").read(u);
        file.getDataSet("module.classifier.1.var").read(v);
        cal_act(act, u, v, scale, shift);
        */
        
        // Read FC Model
        std::vector<std::vector<double>> fc_weight(100, std::vector<double>(512, 0));
        std::vector<double> fc_bias(100, 0.0);
        // Read from pre-trained model
        std::string str = "module.classifier.1.";
        file.getDataSet(str + "weight").read(fc_weight);
        file.getDataSet(str + "bias").read(fc_bias);

        std::cout << "================\n";
        std::cout << "PLAINTEXT DOMAIN\n";
        std::cout << "================\n";
        std::vector<HEaaN::Message> fc_input(512, HEaaN::Message(log_slots));
        std::vector<HEaaN::Message> fc_output(100, HEaaN::Message(log_slots));
#pragma omp parallel for
        for(int i = 0; i < 512; i++) {
            fc_input[i] = output[i][0][0];
        }

        fc_plain(fc_input, fc_output, fc_weight, fc_bias, context, eval);

        std::cout << "=================\n";
        std::cout << "CIPHERTEXT DOMAIN\n";
        std::cout << "=================\n";

        output_cipher.resize(100, HEaaN::Ciphertext(context));
        output_cipher.shrink_to_fit();
        fc_cipher(input_cipher, output_cipher, fc_weight, fc_bias, 100, context, pack, enc, eval, edcoder, dec, sk);
        std::vector<HEaaN::Message> output_msg(100, HEaaN::Message(log_slots));
        for(int i = 0; i < 100; i++) {
            dec.decrypt(output_cipher[i], sk, output_msg[i]);
        }

        int correct_pt = 0, correct_ct = 0;;
        int ans_pt, ans_ct;
        int maxi_pt, maxi_ct;
        for(int b = 0; b < 64; b++) {
            std::cout << "NO " << batch_point + b << " | ";
            ans_pt = 0;
            maxi_pt = fc_output[0][b].real();
            std::cout << "PLAINTEXT > " << fc_output[0][b].real() << " ";
            for(int i = 1; i < 100; i++) {
                std::cout << fc_output[i][b].real() << " ";
                if(fc_output[i][b].real() > maxi_pt) {
                    ans_pt = i;
                    maxi_pt = fc_output[i][b].real();
                }
            }
            std::cout << " | ";

            ans_ct = 0;
            maxi_ct = output_msg[0][b * 512].real();
            std::cout << "CIPHERTEXT > " << output_msg[0][b * 512].real() << " ";
            for(int i = 1; i < 100; i++) {
                std::cout << output_msg[i][b * 512].real() << " ";
                if(output_msg[i][b * 512].real() > maxi_ct) {
                    ans_ct = i;
                    maxi_ct = output_msg[i][b * 512].real();
                }
            }
            std::cout << "ANS > " << ans_pt << " " << ans_ct << " " << label[batch_point + b] << std::endl;
            std::cout << "ANS > " << ans_pt << " " << label[batch_point + b] << std::endl;
            if(ans_pt == label[batch_point + b]) correct_pt++;
            if(ans_ct == label[batch_point + b]) correct_ct++;
        }
        std::cout << "==================================" << std::endl; 
        std::cout << "TOTAL CORRECT PT: " << correct_pt << " CT: " << correct_ct << std::endl; 
        std::cout << "==================================" << std::endl; 
    }
    return 0;
}

void vgg_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
               std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
               std::vector<char> &vgg_meanpool,
               std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b1,
               std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b2,
               std::vector<double> &bias_b1,
               std::vector<double> &bias_b2,
               std::vector<double> &ss,
               std::vector<std::vector<double>> &u,
               std::vector<std::vector<double>> &v,
               bool bitm, 
               HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::cout << "=========\n";
    std::cout << "VGG BLOCK\n";
    std::cout << "=========\n";
    const auto log_slots = getLogFullSlots(context);

    int conv_size_1 = input[0].size();
    int conv_size_2;
    if('M' == vgg_meanpool[0]) {
        conv_size_2 = input[0].size() / 2;
    }else{
        conv_size_2 = conv_size_1;
    }
    int conv_size_pad_1 = conv_size_1 + 2;
    int conv_size_pad_2 = conv_size_2 + 2;
    int kernel_ic_1 = kernel_b1[0].size();
    int kernel_oc_1 = kernel_b1.size();
    int kernel_ic_2 = kernel_b2[0].size();
    int kernel_oc_2 = kernel_b2.size();
    std::vector<double> act(3, 0);

    std::cout << conv_size_1 << " " << conv_size_2 << " " << std::endl;
    std::cout << kernel_ic_1 << " " << kernel_oc_1 << " " << kernel_ic_2 << " " << kernel_oc_2 << std::endl;

    std::vector<std::vector<std::vector<HEaaN::Message>>> input_2(kernel_ic_1, 
                         std::vector<std::vector<HEaaN::Message>>(conv_size_pad_1, 
                                      std::vector<HEaaN::Message>(conv_size_pad_1, 
                                                                  HEaaN::Message(log_slots))));
    // 3*3
    std::cout << "FIRST CONV\n";
    padding_plain(input, input_2, conv_size_1, kernel_ic_1);
    conv_plain(input_2, output, kernel_b1, bias_b1, 1, context, eval);
    cal_act(act, u[0], v[0], ss[0], ss[1]);
    act_plain(output, act, context, eval);

    resize_3d(input, conv_size_1, conv_size_1, kernel_oc_1, context);
    copy_pt(input, output);

    if('M' == vgg_meanpool[0]) {
        // Mean-Pooling
        resize_3d(output, conv_size_2, conv_size_2, kernel_ic_2, context);
        meanpool_plain(input, output, context, eval);
    }

    if(bitm){
        //release memory
        input_2.clear();
        input_2.shrink_to_fit();

        return;
    } 

    // 3*3
    std::cout << "SECOND CONV\n";
    resize_3d(input_2, conv_size_pad_2, conv_size_pad_2, kernel_ic_2, context);
    padding_plain(output, input_2, conv_size_2, kernel_ic_2);
    resize_3d(output, conv_size_2, conv_size_2, kernel_oc_2, context);
    conv_plain(input_2, output, kernel_b2, bias_b2, 1, context, eval);

    cal_act(act, u[1], v[1], ss[2], ss[3]);
    act_plain(output, act, context, eval);

    if('M' == vgg_meanpool[1]) {
        resize_3d(input, conv_size_2, conv_size_2, kernel_oc_2, context);
        copy_pt(input, output);

        conv_size_2 = conv_size_2 / 2;

        // Mean-Pooling
        resize_3d(output, conv_size_2, conv_size_2, kernel_oc_2, context);
        meanpool_plain(input, output, context, eval);
    }


    std::cout << "VGG BLOCK OUTPUT\n";
    for(int oc = 0; oc < kernel_oc_2; oc++) {
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < conv_size_2; i++) {
                for(int j = 0; j < conv_size_2; j++) {
                    std::cout << output[oc][i][j][0].real() << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    //release memory
    input_2.clear();
    input_2.shrink_to_fit();
}

void vgg_cipher(std::vector<HEaaN::Ciphertext> &vec_cipher_1,
                std::vector<HEaaN::Ciphertext> &vec_cipher_2,
                std::vector<char> &vgg_meanpool,
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1,
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2,
                std::vector<double> bias_b1,
                std::vector<double> bias_b2,
                std::vector<double> ss,
                std::vector<std::vector<double>> u,
                std::vector<std::vector<double>> v,
                int input_size, int rp_pack_size, int chnl_batch_size, bool bitm, bool is_gpu,
                HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc,
                HEaaN::Decryptor &dec, HEaaN::SecretKey &sk, HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    std::cout << "=========\n";
    std::cout << "VGG BLOCK\n";
    std::cout << "=========\n";
    const auto log_slots = getLogFullSlots(context);

    int conv_1 = input_size;
    int conv_2;
    if('M' == vgg_meanpool[0]) {
        conv_2 = conv_1 / 2;
    }else{
        conv_2 = conv_1;
    }
    int conv_1_pad = conv_1 + 2;
    int conv_2_pad = conv_2 + 2;
    int kernel_ic_1 = kernel_b1[0].size();
    int kernel_oc_1 = kernel_b1.size();
    int kernel_ic_2 = kernel_b2[0].size();
    int kernel_oc_2 = kernel_b2.size();
    std::vector<double> act(3, 0);

    HEaaN::Message zero(log_slots);
    fillZero(zero);

    std::cout << conv_1 << " " << conv_2 << " " << kernel_oc_1 << " " << kernel_oc_2 << std::endl;
    std::cout << "RP PACK SIZE " << rp_pack_size << std::endl;

    std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1_b(kernel_oc_1 / chnl_batch_size, 
            std::vector<std::vector<std::vector<double>>>(kernel_ic_1, 
                std::vector<std::vector<double>>(3, 
                    std::vector<double>(3, 0.0))));
    std::vector<double> bias_b1_b(kernel_oc_1 / chnl_batch_size, 0.0);
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2_b(kernel_oc_2, 
            std::vector<std::vector<std::vector<double>>>(kernel_ic_2 / chnl_batch_size, 
                std::vector<std::vector<double>>(3, 
                    std::vector<double>(3, 0.0))));
    std::vector<double> bias_b2_b(kernel_oc_2, 0.0);

    std::vector<HEaaN::Ciphertext> vec_cipher_3(conv_1_pad * conv_1_pad, HEaaN::Ciphertext(context));
    std::vector<HEaaN::Ciphertext> agg_cipher(conv_2 * conv_2, HEaaN::Ciphertext(context));

#pragma omp parallel for
    for(int i = 0; i < conv_2 * conv_2; i++) {
        enc.encrypt(zero, pack, agg_cipher[i]);
        eval.levelDown(agg_cipher[i], 4, agg_cipher[i]);
    }

    for(int kocb = 0; kocb < chnl_batch_size; kocb++) {
        std::cout << "================= " << std::endl;
        std::cout << "kernel_oc_batch " << kocb << std::endl;
        std::cout << "================= " << std::endl;
        // slice kernel
        int num_chnl_batch = kernel_oc_1 / chnl_batch_size;
        std::cout << "number of chnl batch: " << num_chnl_batch << std::endl;
        for(int oc = 0; oc < num_chnl_batch; oc++) {
            for(int ic = 0; ic < kernel_ic_1; ic++) {
                for(int i = 0; i < 3; i++) {
                    for(int j = 0; j < 3; j++) {
                        kernel_b1_b[oc][ic][i][j] = kernel_b1[kocb * num_chnl_batch + oc][ic][i][j];
                    }
                }
            }
            bias_b1_b[oc] = bias_b1[kocb * num_chnl_batch + oc];
        }
        for(int oc = 0; oc < kernel_oc_2; oc++) {
            for(int ic = 0; ic < num_chnl_batch; ic++) {
                for(int i = 0; i < 3; i++) {
                    for(int j = 0; j < 3; j++) {
                        kernel_b2_b[oc][ic][i][j] = kernel_b2[oc][kocb * num_chnl_batch + ic][i][j];
                    }
                }
            }
            bias_b2_b[oc] = bias_b2[oc] / chnl_batch_size;
        }

        std::cout << "FIRST CONV\n";
        cal_act(act, u[0], v[0], ss[0], ss[1]);
        if('M' == vgg_meanpool[0]) {
            update_model_mp(kernel_b1_b, bias_b1_b, act);
        }else{
            update_model(kernel_b1_b, bias_b1_b, act);
        }
        padding_cipher(vec_cipher_3, vec_cipher_1, conv_1, 1, context, pack, enc, eval);

        FCA(vec_cipher_3, vec_cipher_2, kernel_b1_b, bias_b1_b, conv_1_pad, conv_1_pad, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);
        act_cipher(vec_cipher_2, act, conv_1, conv_1, context, edcoder, eval);
        if('M' == vgg_meanpool[0]) {
            meanpool_cipher(vec_cipher_2, vec_cipher_3, conv_1, conv_1, context, pack, enc, eval);
            copy_ct(vec_cipher_2, vec_cipher_3, conv_2, conv_2);
        }

        IRB2A(vec_cipher_2, conv_2, conv_2, rp_pack_size, num_chnl_batch, kernel_oc_2, is_gpu, context, pack, enc, eval, edcoder);
        print_ct(vec_cipher_2, conv_2, conv_2, rp_pack_size, num_chnl_batch, "B", dec, sk);

        if(!bitm) {
            std::cout << "SECOND CONV\n";
            cal_act(act, u[1], v[1], ss[2], ss[3]);
            if('M' == vgg_meanpool[1]) {
                update_model_mp(kernel_b2_b, bias_b2_b, act);
            }else{
                update_model(kernel_b2_b, bias_b2_b, act);
            }
            padding_cipher(vec_cipher_3, vec_cipher_2, conv_2, 1, context, pack, enc, eval);

            FCA(vec_cipher_3, vec_cipher_2, kernel_b2_b, bias_b2_b, conv_2_pad, conv_2_pad, rp_pack_size, true, "A", is_gpu, context, pack, enc, eval, edcoder);
        }else{
            std::cout << "LEVEL " << vec_cipher_2[0].getLevel() << std::endl;
            A2B(vec_cipher_2, conv_2, conv_2, num_chnl_batch, rp_pack_size, is_gpu, context, pack, enc, eval, edcoder);

            HEaaN::Message filter_msg(log_slots);
            fillZero(filter_msg);
            for(int i = 0; i < num_chnl_batch; i++) {
                filter_msg[i].real(1);
            }
            HEaaN::Plaintext filter_plain(context);
            filter_plain = edcoder.encode(filter_msg);
            filter_plain.setLevel(vec_cipher_2[0].getLevel());

#pragma omp parallel for
            for(int i = 0; i < conv_2 * conv_2; i++) {
                eval.mult(vec_cipher_2[i], filter_plain, vec_cipher_2[i]);
                eval.rightRotate(vec_cipher_2[i], 64 * kocb, vec_cipher_2[i]);
            }
            std::cout << "LEVEL END " << vec_cipher_2[0].getLevel() << std::endl;
        }

        // aggregrate
#pragma omp parallel for
        for(int i = 0; i < conv_2 * conv_2; i++) {
            eval.add(agg_cipher[i], vec_cipher_2[i], agg_cipher[i]);
        }
    }

    // distribute
#pragma omp parallel for
    for(int i = 0; i < conv_2 * conv_2; i++) {
        vec_cipher_2[i] = agg_cipher[i];
    }

    /*
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    HEaaN::Message dmsg;
    for(int i = 0; i < conv_2; i++) {
        for(int j = 0; j < conv_2; j++) {
            int output_pos = i * conv_2 + j;
            dec.decrypt(vec_cipher_2[output_pos], sk, dmsg);
            for(int ii = 0; ii < 257; ii++) {
                std::cout << dmsg[ii].real() << " " ;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    */

    // act_cipher(vec_cipher_2, act, conv_2, conv_2, context, edcoder, eval);
    // print_ct(vec_cipher_2, conv_2, conv_2, rp_pack_size, kernel_oc_2, "A", dec, sk);

    // release memory
    vec_cipher_3.clear();
    vec_cipher_3.shrink_to_fit();
}

