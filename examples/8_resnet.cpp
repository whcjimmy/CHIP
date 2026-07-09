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

void resnet_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                  std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b1,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b2,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b3,
                  std::vector<double> &bias_b1,
                  std::vector<double> &bias_b2,
                  std::vector<double> &bias_b3,
                  std::vector<double> &ss,
                  std::vector<std::vector<double>> &u,
                  std::vector<std::vector<double>> &v,
                  std::vector<std::vector<double>> &ds_bn,
                  int stride,
                  HEaaN::Context &context, HEaaN::HomEvaluator &eval);

void resnet_cipher(std::vector<HEaaN::Ciphertext> &vec_cipher_1,
                   std::vector<HEaaN::Ciphertext> &vec_cipher_2,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b3,
                   std::vector<double> bias_b1,
                   std::vector<double> bias_b2,
                   std::vector<double> bias_b3,
                   std::vector<double> ss,
                   std::vector<std::vector<double>> u,
                   std::vector<std::vector<double>> v,
                   std::vector<std::vector<double>> &ds_bn,
                   int input_size, int stride, 
                   bool rp_last, int rp_pack_size, bool is_gpu,
                   HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc,
                   HEaaN::Decryptor &dec, HEaaN::SecretKey &sk, HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder);

int main(int argc, char *argv[]) {
    omp_set_num_threads(40);
    int stride       = 1;
    int kernel_ic    = 3;
    int kernel_oc    = 16;
    int kernel_row   = 3;
    int kernel_col   = 3;
    int input_row    = 34;
    int input_col    = 34;
    int is_gpu       = false;
    std::string type = "A";
    int batch_point  = atoi(argv[1]);
    std::cout << "Batch starts from No." << batch_point << std::endl;
    int icp = pow(2, ceil(log2(kernel_ic)));
    int dca_output_row = int(ceil(1.0 * (input_row - kernel_row + 1) / stride));
    int dca_output_col = int(ceil(1.0 * (input_col - kernel_col + 1) / stride));

    std::cout << "========\n";
    if(is_gpu) {
        std::cout << "GPU MODE\n";
    }else{
        std::cout << "CPU MODE\n";
    }
    std::cout << "========\n";

    std::cout << "OUTPUT SIZE " << dca_output_row << " " << dca_output_col << std::endl;

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
        std::cout << "================\n";
        std::cout << "PLAINTEXT DOMAIN\n";
        std::cout << "================\n";
        std::vector<std::vector<std::vector<std::vector<double>>>> input_double(10000, 
                                  std::vector<std::vector<std::vector<double>>>(kernel_ic, 
                                               std::vector<std::vector<double>>(input_row - 2, 
                                                            std::vector<double>(input_col - 2, 0.0))));
        std::vector<double> label(10000);
        std::vector<std::vector<std::vector<HEaaN::Message>>> input(kernel_ic, 
                           std::vector<std::vector<HEaaN::Message>>(input_row, 
                                        std::vector<HEaaN::Message>(input_col, 
                                                                    HEaaN::Message(log_slots))));
        std::vector<std::vector<std::vector<HEaaN::Message>>> output(kernel_oc, 
                            std::vector<std::vector<HEaaN::Message>>(dca_output_row, 
                                         std::vector<HEaaN::Message>(dca_output_col, 
                                                                     HEaaN::Message(log_slots))));
        std::vector<std::vector<std::vector<std::vector<double>>>> kernel(kernel_oc, 
                            std::vector<std::vector<std::vector<double>>>(kernel_ic, 
                                         std::vector<std::vector<double>>(kernel_row, 
                                                      std::vector<double>(kernel_col, 0.0))));
        std::vector<double> bias(kernel_oc, 0.0);
        std::vector<double> act(3);
        double scale, shift;
        std::vector<double>u(2);
        std::vector<double>v(2);

        /*
        generate_3D_data(input);
        generate_kernel_bias(kernel, bias);
        */

        // Open the HDF5 MODEL file
        HighFive::File file("../../model/c10_rn20_9264.h5", HighFive::File::ReadOnly);
        // HighFive::File file("../model/c10_rn32_9218.h5", HighFive::File::ReadOnly);
        // Read dataset
        file.getDataSet("module.conv1.weight").read(kernel);
        // file.getDataSet("moduleconv1.bias").read(bias);
        generate_1D_data_double(bias);
        file.getDataSet("module.herpn.scale").read(scale);
        file.getDataSet("module.herpn.shift").read(shift);
        file.getDataSet("module.herpn.mean").read(u);
        file.getDataSet("module.herpn.var").read(v);

        // Open the HDF5 INPUT FILE
        // Read input and label
        HighFive::File file_2("../../model/input_c10.h5", HighFive::File::ReadOnly);
        file_2.getDataSet("input").read(input_double);
        file_2.getDataSet("label").read(label);
        for(int b = 0; b < 512; b++) {
            for(int ic = 0; ic < 3; ic++) {
                for(int i = 0; i < 34; i++) {
                    for(int j = 0; j < 34; j++) {
                        if(i == 0 || j == 0 || i == 33 || j == 33) {
                            fillZero(input[ic][i][j]);
                        }else{
                            input[ic][i][j][b].real(input_double[batch_point + b][ic][i - 1][j - 1]);
                            input[ic][i][j][b].imag(0);
                            // fillRandomComplexKernel(input[ic][i][j], input_double[b][ic][i - 1][j - 1]);
                        }
                    }
                }
            }
        }

        // First Conv
        conv_plain(input, output, kernel, bias, stride, context, eval);
        cal_act(act, u, v, scale, shift);
        act_plain(output, act, context, eval);

        std::cout << "=================\n";
        std::cout << "ENCRYPTION DOMAIN\n";
        std::cout << "=================\n";
        process_mem_usage(vm, rss);
        std::vector<HEaaN::Ciphertext> input_cipher(input_row * input_col, HEaaN::Ciphertext(context));
        std::vector<HEaaN::Ciphertext> output_cipher(dca_output_row * dca_output_col, HEaaN::Ciphertext(context));

        // ENCRYPT INPUT
#pragma omp parallel for collapse(2)
        for(int i = 0; i < input_row; i++) {
            for(int j = 0; j < input_col; j++) {
                HEaaN::Message msg(log_slots);
                fillZero(msg);
                for(int b = 0; b < 512; b++) {
                    for(int oc = 0; oc < kernel_oc; oc++) {
                        for(int ic = 0; ic < icp; ic++) {
                            int pos = (type == "A") ?  
                                pos = b * kernel_oc * icp + ic * kernel_oc + oc : 
                                pos = b * kernel_oc * icp + oc * icp + ic;
                            if(ic < kernel_ic) {
                               msg[pos] = input[ic][i][j][b];
                            }else{
                                msg[pos].real(0);
                            }
                        }
                    }
                }
                int enc_pos = i * input_col + j;
                enc.encrypt(msg, pack, input_cipher[enc_pos]);
                eval.levelDown(input_cipher[enc_pos], 11, input_cipher[enc_pos]);
            }
        }
        process_mem_usage(vm2, rss2);
        std::cout << "INPUT and OUTPUT CIPHER RAM USAGE " << (vm2 - vm)/(1024 * 1024) << " GB" << std::endl;
        std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;
        std::cout << "input_cipher level " << input_cipher[0].getLevel() << std::endl;

        /*
        ofstream ofs("ciphertext.dat", ios::binary);
        for(int i = 0; i < input_row; i++) {
            for(int j = 0; j < input_col; j++) {
                input_cipher[i * input_col + j].save(ofs);
            }
        }
        ofs.close();
        */

        // First Conv
        update_model(kernel, bias, act);

        // DCA(input_cipher, output_cipher, kernel, bias, input_row, input_col, 1, true, type, is_gpu, context, pack, enc, eval, edcoder);

        if(stride == 1) {
            if(kernel_row == 5) {
                // pg_rt.FCA_5_5(pp, "pt", input_cipher, kernel, bias, output_cipher, input_row, input_col, 4);
            }else{
                FCA(input_cipher, output_cipher, kernel, bias, input_row, input_col, icp, true, type, is_gpu, context, pack, enc, eval, edcoder);
            }
        }else if (stride == 2) {
            if(kernel_row == 3) {
                FCA_s2_3_3(input_cipher, kernel, bias, output_cipher, input_row, input_col, icp, type, is_gpu, context, pack, enc, eval, edcoder);
            }else if(kernel_row == 5) {
                // pg_rt.FCA_s2_5_5(pp, "pt", input_cipher, kernel, bias, output_cipher, output_pile_row, output_pile_col, input_row, input_col, 5);
            }
        }

        process_mem_usage(vm2, rss2);
        std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;

        act_cipher(output_cipher, act, dca_output_row, dca_output_col, context, edcoder, eval);
        copy_ct(input_cipher, output_cipher, dca_output_row, dca_output_col);
        print_ct(input_cipher, dca_output_row, dca_output_col, kernel_ic, kernel_oc, type, dec, sk);

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

        resize_3d(input, dca_output_row, dca_output_col, kernel_oc, context);
        copy_pt(input, output);

        // Clean Memory
        input_double.clear();
        input_double.shrink_to_fit();
        kernel.clear();
        kernel.shrink_to_fit();
        bias.clear();
        bias.shrink_to_fit();
        act.clear();
        act.shrink_to_fit();
        u.clear();
        u.shrink_to_fit();
        v.clear();
        v.shrink_to_fit();

        // RESNET
        int resnet_stride = 1;
        int resnet_ic = kernel_oc;
        int resnet_oc = kernel_oc;
        int resnet_input_size = dca_output_row; 
        int resnet_output_size = dca_output_row;
        int rp_pack_size = kernel_ic;
        int rp_times = 3; // 3 for RESNET-20 and 5 for RESNET-32
        bool rp_last = false;
        int batch_size = (resnet_oc * resnet_oc) * 512 / 32768;
        
        std::vector<HEaaN::Ciphertext> input_cipher_2(resnet_input_size * resnet_input_size, HEaaN::Ciphertext(context));
        std::vector<HEaaN::Ciphertext> output_cipher_2(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));

        if(is_gpu) {
            std::cout << "Load boot constants data to GPU memory ..." << std::endl;
            boot.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
            // eval.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
        }

        HEaaN::Message zero(log_slots);
        fillZero(zero);

        for(int rnl = 0; rnl < 3; rnl++) {
            for(int rp = 0; rp < rp_times; rp++) {
                std::cout << "================================\n";
                std::cout << "================================\n\n";
                std::cout << "RENSET LAYER " << rnl << " REPEAT " << rp << "-th TIMES\n\n";
                std::cout << "================================\n";
                std::cout << "================================\n";

                (rnl != 0 && rp == 0) ? resnet_stride = 2 : resnet_stride = 1;

                resnet_input_size = resnet_output_size;
                resnet_output_size = resnet_output_size / resnet_stride;

                resnet_ic = resnet_oc;
                resnet_oc = resnet_oc * resnet_stride;

                (rp == rp_times - 1) ? rp_last = true : rp_last = false;
                if(rnl == 0 || rnl == 2) rp_last = false;
                rp_pack_size = resnet_oc;
                batch_size = (resnet_oc * resnet_oc) * 512 / 32768;

                std::cout << resnet_input_size << " " << resnet_output_size << " " << resnet_ic << " " << resnet_oc << " " << resnet_stride << std::endl;
                std::cout << " " << rp_last << " " << rp_pack_size << " " << batch_size << std::endl;

                // prepare weight and bias
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1(resnet_oc, 
                        std::vector<std::vector<std::vector<double>>>(resnet_ic, 
                            std::vector<std::vector<double>>(3, 
                                std::vector<double>(3, 0.0))));
                std::vector<double> bias_b1(resnet_oc, 0.0);
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2(resnet_oc, 
                        std::vector<std::vector<std::vector<double>>>(resnet_oc, 
                            std::vector<std::vector<double>>(3, 
                                std::vector<double>(3, 0.0))));
                std::vector<double> bias_b2(resnet_oc, 0.0);
                std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b3(resnet_oc, 
                        std::vector<std::vector<std::vector<double>>>(resnet_ic, 
                            std::vector<std::vector<double>>(1, 
                                std::vector<double>(1, 0.0))));
                std::vector<double> bias_b3(resnet_oc, 0.0);
                std::vector<std::vector<double>> ds_bn(4, std::vector<double>(resnet_oc, 0.0));

                std::vector<double> ss(4);
                std::vector<std::vector<double>>u(2, std::vector<double>(2));
                std::vector<std::vector<double>>v(2, std::vector<double>(2));

                // Read from pre-trained model
                std::string str = "module.res" + std::to_string(rnl+1) + "." + std::to_string(rp) + ".";
                file.getDataSet(str + "conv1.weight").read(kernel_b1);
                // file.getDataSet(str + "conv1.bias").read(bias_b1);
                generate_1D_data_double(bias_b1);
                file.getDataSet(str + "conv2.weight").read(kernel_b2);
                // file.getDataSet(str + "residual.2.bias").read(bias_b2);
                generate_1D_data_double(bias_b2);
                if(resnet_stride == 2) {
                    file.getDataSet(str + "ds.0.weight").read(kernel_b3);
                    // file.getDataSet(str + "shortcut.bias").read(bias_b3);
                    generate_1D_data_double(bias_b3);
                    file.getDataSet(str + "ds.1.weight").read(ds_bn[0]);
                    file.getDataSet(str + "ds.1.bias").read(ds_bn[1]);
                    file.getDataSet(str + "ds.1.running_mean").read(ds_bn[2]);
                    file.getDataSet(str + "ds.1.running_var").read(ds_bn[3]);
                }
                file.getDataSet(str + "herpn_1.scale").read(ss[0]);
                file.getDataSet(str + "herpn_1.shift").read(ss[1]);
                file.getDataSet(str + "herpn_1.mean").read(u[0]);
                file.getDataSet(str + "herpn_1.var").read(v[0]);
                file.getDataSet(str + "herpn_2.scale").read(ss[2]);
                file.getDataSet(str + "herpn_2.shift").read(ss[3]);
                file.getDataSet(str + "herpn_2.mean").read(u[1]);
                file.getDataSet(str + "herpn_2.var").read(v[1]);

                std::cout << "================\n";
                std::cout << "PLAINTEXT DOMAIN\n";
                std::cout << "================\n";

                resize_3d(output, resnet_output_size, resnet_output_size, resnet_oc, context);

                resnet_plain(input, output, kernel_b1, kernel_b2, kernel_b3, bias_b1, bias_b2, bias_b3, ss, u, v, ds_bn, resnet_stride, context, eval);

                resize_3d(input, resnet_output_size, resnet_output_size, resnet_oc, context);
                copy_pt(input, output);

                std::cout << "=================\n";
                std::cout << "ENCRYPTION DOMAIN\n";
                std::cout << "=================\n";

                output_cipher.resize(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));
                output_cipher.shrink_to_fit();
                output_cipher_2.resize(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));
                output_cipher_2.shrink_to_fit();
#pragma omp parallel for
                for(int i = 0; i < resnet_output_size * resnet_output_size; i++) {
                    enc.encrypt(zero, pack, output_cipher[i]);
                    eval.levelDown(output_cipher[i], 3, output_cipher[i]);
                }

                for(int b = 0; b < batch_size; b++) {
                // for(int b = 0; b < 1; b++) {
                    std::cout << "============= " << std::endl;
                    std::cout << "BATCH SIZE " << b << std::endl;
                    std::cout << "============= " << std::endl;
                    // SPLIT (1->many)
                    SPLIT(input_cipher, input_cipher_2, resnet_input_size, resnet_input_size, resnet_oc, rp_pack_size, b * 64, "A", is_gpu, context, pack, enc, eval, edcoder);

                    resnet_cipher(input_cipher_2, output_cipher_2, kernel_b1, kernel_b2, kernel_b3, bias_b1, bias_b2, bias_b3, 
                            ss, u, v, ds_bn, resnet_input_size, resnet_stride, rp_last, rp_pack_size, is_gpu, context, pack, enc, dec, sk, eval, edcoder);

                    // MERGE (many->1)
                    MERGE(output_cipher_2, output_cipher, resnet_output_size, resnet_output_size, resnet_oc, rp_pack_size, b * 64, "A", is_gpu, context, pack, enc, eval, edcoder);
                }

                input_cipher.resize(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));
                input_cipher.shrink_to_fit();
                input_cipher_2.resize(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));
                input_cipher_2.shrink_to_fit();
                copy_ct(input_cipher, output_cipher, resnet_output_size, resnet_output_size);

                std::cout << "BOOTSTRAPPING\n";
                HEaaN::HEaaNTimer timer(true);
                timer.start("BOOTSTRAPPING ");
#pragma omp parallel for if(!is_gpu)
                for(int i = 0; i < resnet_output_size * resnet_output_size; i++) {
                    // eval.bootstrap(input_cipher[i], input_cipher[i]);
                    boot.bootstrap(input_cipher[i], input_cipher[i]);
                    eval.levelDown(input_cipher[i], 10, input_cipher[i]);
                }
                timer.end();
                timer.print();

                std::vector<double> act(3);
                cal_act(act, u[1], v[1], ss[2], ss[3]);
                act[1] = act[1] / sqrt(act[0]);
                act_cipher(input_cipher, act, resnet_output_size, resnet_output_size, context, edcoder, eval);
            }
        }

        // AVGPOOL + FC
        std::vector<std::vector<double>> fc_weight(10, std::vector<double>(64, 0));
        std::vector<double> fc_bias(10, 0.0);
        std::vector<HEaaN::Message> fc_output(10, HEaaN::Message(log_slots));
        // Read from pre-trained model
        file.getDataSet("module.fc.weight").read(fc_weight);
        file.getDataSet("module.fc.bias").read(fc_bias);
        // generate_1D_data_double(fc_bias);

        avgpool_fc_plain(input, fc_output, fc_weight, fc_bias, context, eval);

        output_cipher.resize(10, HEaaN::Ciphertext(context));
        output_cipher.shrink_to_fit();
        avgpool_fc_cipher(input_cipher, output_cipher, fc_weight, fc_bias, context, pack, enc, eval, edcoder, dec, sk);
        std::vector<HEaaN::Message> output_msg(10, HEaaN::Message(log_slots));
        for(int i = 0; i < 10; i++) {
            dec.decrypt(output_cipher[i], sk, output_msg[i]);
        }
        int correct_pt = 0, correct_ct = 0;;
        int ans_pt, ans_ct;
        int maxi_pt, maxi_ct;
        for(int b = 0; b < 512; b++) {
            std::cout << "NO " << batch_point + b << " | ";
            ans_pt = 0;
            maxi_pt = fc_output[0][b].real();
            std::cout << "PLAINTEXT > " << fc_output[0][b].real() << " ";
            for(int i = 1; i < 10; i++) {
                std::cout << fc_output[i][b].real() << " ";
                if(fc_output[i][b].real() > maxi_pt) {
                    ans_pt = i;
                    maxi_pt = fc_output[i][b].real();
                }
            }
            std::cout << " | ";

            ans_ct = 0;
            maxi_ct = output_msg[0][b * 64].real();
            std::cout << "CIPHERTEXT > " << output_msg[0][b * 64].real() << " ";
            for(int i = 1; i < 10; i++) {
                std::cout << output_msg[i][b * 64].real() << " ";
                if(output_msg[i][b * 64].real() > maxi_ct) {
                    ans_ct = i;
                    maxi_ct = output_msg[i][b * 64].real();
                }
            }
            std::cout << "ANS > " << ans_pt << " " << ans_ct << " " << label[batch_point + b] << std::endl;
            if(ans_pt == label[batch_point + b]) correct_pt++;
            if(ans_ct == label[batch_point + b]) correct_ct++;
        }
        std::cout << "==================================" << std::endl; 
        std::cout << "TOTAL CORRECT PT: " << correct_pt << " CT: " << correct_ct << std::endl; 
        std::cout << "==================================" << std::endl; 

        std::cout << "==================================" << std::endl; 
        std::cout << "TOTAL TOTAL_TIME " << total_total_time << std::endl;
        std::cout << "==================================" << std::endl; 

    }
    return 0;
}

void resnet_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                  std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b1,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b2,
                  std::vector<std::vector<std::vector<std::vector<double>>>> &kernel_b3,
                  std::vector<double> &bias_b1,
                  std::vector<double> &bias_b2,
                  std::vector<double> &bias_b3,
                  std::vector<double> &ss,
                  std::vector<std::vector<double>> &u,
                  std::vector<std::vector<double>> &v,
                  std::vector<std::vector<double>> &ds_bn,
                  int stride, 
                  HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::cout << "============\n";
    std::cout << "RESNET BLOCK\n";
    std::cout << "============\n";
    const auto log_slots = getLogFullSlots(context);

    int input_size  = input[0].size();
    int output_size = output[0].size();
    int conv1_size;
    if(stride == 1) {
        conv1_size = input_size + 2;
    }else{
        conv1_size = input_size + 1;
    }
    int conv2_size = output_size + 2;
    int kernel_ic  = kernel_b1[0].size();
    int kernel_oc  = kernel_b1.size();
    std::vector<double> act(3, 0);

    std::cout << input_size << " " << output_size << " " << kernel_ic << " " << kernel_oc << std::endl;

    std::vector<std::vector<std::vector<HEaaN::Message>>> input_2(kernel_ic, 
                         std::vector<std::vector<HEaaN::Message>>(conv1_size, 
                                      std::vector<HEaaN::Message>(conv1_size, 
                                                                  HEaaN::Message(log_slots))));
    // 3*3
    std::cout << "FIRST CONV\n";
    padding_plain(input, input_2, input_size, kernel_ic);
    conv_plain(input_2, output, kernel_b1, bias_b1, stride, context, eval);
    cal_act(act, u[0], v[0], ss[0], ss[1]);
    act_plain(output, act, context, eval);

    // 3*3
    std::cout << "SECOND CONV\n";
    resize_3d(input_2, conv2_size, conv2_size, kernel_oc, context);
    padding_plain(output, input_2, output_size, kernel_oc);
    conv_plain(input_2, output, kernel_b2, bias_b2, 1, context, eval);

    // shortcut
    if(stride == 2) {
        // 1*1
        std::cout << "THIRD CONV\n";
        resize_3d(input_2, output_size, output_size, kernel_oc, context);
        conv_plain(input, input_2, kernel_b3, bias_b3, stride, context, eval);

        // BN
#pragma omp parallel for collapse(3)
        for(int oc = 0; oc < input_2.size(); oc++) {
            for(int i = 0; i < input_2[0].size(); i++) {
                for(int j = 0; j < input_2[0][0].size(); j++) {
                    
                    std::vector<HEaaN::Message> bn_msg(4, HEaaN::Message(log_slots));
                    fillRandomComplexKernel(bn_msg[0], ds_bn[0][oc]); // weight
                    fillRandomComplexKernel(bn_msg[1], ds_bn[1][oc]); //bias
                    fillRandomComplexKernel(bn_msg[2], ds_bn[2][oc]); // running_mean
                    fillRandomComplexKernel(bn_msg[3], 1 / sqrt(ds_bn[3][oc] + 0.00001)); // running_var

                    eval.sub( input_2[oc][i][j], bn_msg[2], input_2[oc][i][j]);
                    eval.mult(input_2[oc][i][j], bn_msg[3], input_2[oc][i][j]);
                    eval.mult(input_2[oc][i][j], bn_msg[0], input_2[oc][i][j]);
                    eval.add( input_2[oc][i][j], bn_msg[1], input_2[oc][i][j]);
                }
            }
        }

        // ADD
        std::cout << "ADD\n";
#pragma omp parallel for collapse(3)
        for(int oc = 0; oc < kernel_oc; oc++) {
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
                    msgAdd(output[oc][i][j], input_2[oc][i][j]);
                }
            }
        }
    }else{
        std::cout << "SHORTCUT\n";
#pragma omp parallel for collapse(3)
        for(int oc = 0; oc < kernel_oc; oc++) {
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
                    msgAdd(output[oc][i][j], input[oc][i][j]);
                }
            }
        }
    }

    cal_act(act, u[1], v[1], ss[2], ss[3]);
    act_plain(output, act, context, eval);

    std::cout << "RESNET BLOCK OUTPUT\n";
    for(int oc = 0; oc < kernel_oc; oc++) {
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
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

void resnet_cipher(std::vector<HEaaN::Ciphertext> &vec_cipher_1,
                   std::vector<HEaaN::Ciphertext> &vec_cipher_2,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2,
                   std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b3,
                   std::vector<double> bias_b1,
                   std::vector<double> bias_b2,
                   std::vector<double> bias_b3,
                   std::vector<double> ss,
                   std::vector<std::vector<double>> u,
                   std::vector<std::vector<double>> v,
                   std::vector<std::vector<double>> &ds_bn,
                   int input_size, int stride, 
                   bool rp_last, int rp_pack_size, bool is_gpu,
                   HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc,
                   HEaaN::Decryptor &dec, HEaaN::SecretKey &sk, HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    std::cout << "============\n";
    std::cout << "RESNET BLOCK\n";
    std::cout << "============\n";
    const auto log_slots = getLogFullSlots(context);

    int conv1_size = (stride == 1) ? input_size + 2 : input_size + 1;
    int kernel_size = kernel_b1[0][0].size();
    int output_size = int(ceil(1.0 * (input_size + 2 - kernel_size + 1) / stride));
    int conv2_size = output_size + 2;
    int resnet_ic   = kernel_b1[0].size();
    int resnet_oc   = kernel_b1.size();
    // int resnet_ic_n = resnet_oc;
    // int resnet_oc_n = (rp_last == true) ? resnet_oc * 2 : resnet_oc;
    std::vector<double> act(3);

    std::cout << input_size << " " << output_size << " " << resnet_ic << " " << resnet_oc << std::endl;
    std::cout << "RP PACK SIZE " << rp_pack_size << std::endl;

    std::vector<HEaaN::Ciphertext> vec_cipher_3;
    for(int i = 0; i < conv1_size * conv1_size; i++) {
        vec_cipher_3.push_back(HEaaN::Ciphertext(context));
    }

    std::vector<HEaaN::Ciphertext> input_cipher_2;
    if(stride == 1) {
        for(int i = 0; i < input_size * input_size; i++) {
            input_cipher_2.push_back(vec_cipher_1[i]);
        }
    }

    IRA2B(vec_cipher_1, input_size, input_size, resnet_oc, resnet_oc, is_gpu, context, pack, enc, eval, edcoder);
    print_ct(vec_cipher_1, input_size, input_size, rp_pack_size, resnet_oc, "B", dec, sk);

    // 3*3
    std::cout << "FIRST CONV\n";
    cal_act(act, u[0], v[0], ss[0], ss[1]);
    update_model(kernel_b1, bias_b1, act);
    padding_cipher(vec_cipher_3, vec_cipher_1, input_size, stride, context, pack, enc, eval);

    if(stride == 1) {
        FCA(vec_cipher_3, vec_cipher_2, kernel_b1, bias_b1, input_size + 2, input_size + 2, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);
    }else{
        FCA_s2_3_3(vec_cipher_3, kernel_b1, bias_b1, vec_cipher_2, input_size + 1, input_size + 1, rp_pack_size, "B", is_gpu, context, pack, enc, eval, edcoder);
    }
    IRB2A(vec_cipher_2, output_size, output_size, rp_pack_size, resnet_oc, resnet_oc, is_gpu, context, pack, enc, eval, edcoder);
    act_cipher(vec_cipher_2, act, output_size, output_size, context, edcoder, eval);
    print_ct(vec_cipher_2, output_size, output_size, rp_pack_size, resnet_oc, "A", dec, sk);

    /*
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    HEaaN::Message dmsg;
    std::string the_type = "A";
    int icp = pow(2, ceil(log2(resnet_ic)));
    for(int oc = 0; oc < resnet_oc; oc++){
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
                    int output_pos = i * output_size + j;
                    int chnl_pos = (the_type == "A") ? oc : oc * icp; 
                    dec.decrypt(vec_cipher_2[output_pos], sk, dmsg);
                    std::cout << dmsg[chnl_pos].real() << " " ;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    */
    
    // 3*3
    std::cout << "SECOND CONV\n";
    vec_cipher_3.resize(conv2_size * conv2_size, HEaaN::Ciphertext(context));
    vec_cipher_3.shrink_to_fit();

    cal_act(act, u[1], v[1], ss[2], ss[3]);
    update_model(kernel_b2, bias_b2, act);
    padding_cipher(vec_cipher_3, vec_cipher_2, output_size, 1, context, pack, enc, eval);

    FCA(vec_cipher_3, vec_cipher_2, kernel_b2, bias_b2, conv2_size, conv2_size, rp_pack_size, true, "A", is_gpu, context, pack, enc, eval, edcoder);

    /*
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    HEaaN::Message dmsg;
    std::string the_type = "B";
    int icp = pow(2, ceil(log2(resnet_oc)));
    for(int oc = 0; oc < resnet_oc; oc++){
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
                    int output_pos = i * output_size + j;
                    int chnl_pos = (the_type == "A") ? oc : oc * icp; 
                    dec.decrypt(vec_cipher_2[output_pos], sk, dmsg);
                    std::cout << dmsg[chnl_pos].real() << " " ;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
    */
    
    // shortcut
    if(stride == 2) {
        int level = vec_cipher_1[0].getLevel();
#pragma omp parallel for
        for(int i = 0; i < input_size * input_size; i++) {
            eval.levelDown(vec_cipher_1[i], level - 1, vec_cipher_1[i]);
        }

        // 1*1
        std::cout << "THIRD CONV\n";
        vec_cipher_3.resize(output_size * output_size, HEaaN::Ciphertext(context));
        vec_cipher_3.shrink_to_fit();

        cal_act(act, u[1], v[1], ss[2], ss[3]);
        update_model_bn(kernel_b3, bias_b3, act, ds_bn);

        DCA(vec_cipher_1, vec_cipher_3, kernel_b3, bias_b3, input_size, input_size, stride, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);
        /*
        std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
        HEaaN::Message dmsg;
        std::string the_type = "B";
        int icp = pow(2, ceil(log2(resnet_ic)));
        for(int oc = 0; oc < resnet_oc; oc++){
            if(oc == 0) {
                std::cout << "oc " << oc << std::endl;
                for(int i = 0; i < output_size; i++) {
                    for(int j = 0; j < output_size; j++) {
                        int output_pos = i * output_size + j;
                        int chnl_pos = (the_type == "A") ? oc : oc * icp; 
                        dec.decrypt(vec_cipher_3[output_pos], sk, dmsg);
                        std::cout << dmsg[chnl_pos].real() << " " ;
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << "INSIDE RESNET CIPHER OUTPUT\n";
        */
        IRB2A(vec_cipher_3, output_size, output_size, rp_pack_size, resnet_oc, resnet_oc, is_gpu, context, pack, enc, eval, edcoder);
        A2B(vec_cipher_3, output_size, output_size, resnet_oc, rp_pack_size, is_gpu, context, pack, enc, eval, edcoder);

#pragma omp parallel for
        for(int i = 0; i < output_size * output_size; i++) {
            eval.add(vec_cipher_2[i], vec_cipher_3[i], vec_cipher_2[i]);
        }
    }else{
        std::cout << "SHORTCUT\n";
        cal_act(act, u[1], v[1], ss[2], ss[3]);
        HEaaN::Plaintext act_0_plain(context);
        HEaaN::Message act_0_msg(log_slots);
        fillRandomComplexKernel(act_0_msg, sqrt(act[0]));
        act_0_plain = edcoder.encode(act_0_msg);
        act_0_plain.setLevel(vec_cipher_2[0].getLevel() + 1);
#pragma omp parallel for
        for(int i = 0; i < input_size * input_size; i++) {
            eval.levelDown(input_cipher_2[i], vec_cipher_2[0].getLevel() + 1, input_cipher_2[i]);
            eval.mult(input_cipher_2[i], act_0_plain, input_cipher_2[i]);
        }
#pragma omp parallel for
        for(int i = 0; i < output_size * output_size; i++) {
            eval.add(vec_cipher_2[i], input_cipher_2[i], vec_cipher_2[i]);
        }
    }

    // release memory
    vec_cipher_3.clear();
    vec_cipher_3.shrink_to_fit();
    input_cipher_2.clear();
    input_cipher_2.shrink_to_fit();
}

