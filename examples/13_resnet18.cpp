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
                   int rp_pack_size, int chnl_batch_size, int split_chnl_size, bool is_gpu,
                   HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc,
                   HEaaN::Decryptor &dec, HEaaN::SecretKey &sk, HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder);


int main(int argc, char *argv[]) {
    omp_set_num_threads(40);
    int is_gpu       = false;
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
        HighFive::File file("../../model/c10_r18_9528.h5", HighFive::File::ReadOnly);
        // Read dataset
        file.getDataSet("module.conv1.weight").read(kernel);
        // file.getDataSet("moduleconv1.bias").read(bias);
        generate_1D_data_double(bias);
        file.getDataSet("module.herpn.scale").read(scale);
        file.getDataSet("module.herpn.shift").read(shift);
        file.getDataSet("module.herpn.mean").read(u);
        file.getDataSet("module.herpn.var").read(v);

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

        // RESNET
        int resnet_stride = 1;
        int resnet_ic = 64;
        int resnet_oc = 64;
        int resnet_input_size = 32; 
        int resnet_output_size = 32;
        int rp_pack_size = 4;
        int rp_times = 2;
        int batch_size;
        int chnl_batch_size; // the num of batch to process the whole num of channels
        int split_chnl_size; // the num of input chanels each batch can process
        
        std::vector<HEaaN::Ciphertext> input_cipher_2(resnet_input_size * resnet_input_size, HEaaN::Ciphertext(context));
        std::vector<HEaaN::Ciphertext> output_cipher_2(resnet_output_size * resnet_output_size, HEaaN::Ciphertext(context));

        if(is_gpu) {
            std::cout << "Load boot constants data to GPU memory ..." << std::endl;
            boot.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
            // eval.loadBootConstants(log_slots, HEaaN::DeviceType::GPU);
        }

        HEaaN::Message zero(log_slots);
        fillZero(zero);

        for(int rnl = 0; rnl < 4; rnl++) {
            for(int rp = 0; rp < 2; rp++) {
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

                rp_pack_size = resnet_oc;

                if(rp_pack_size * resnet_oc <= 32768) {
                    batch_size = 64/(32768/(rp_pack_size * resnet_oc));
                    chnl_batch_size = 1;
                    split_chnl_size = resnet_oc;
                }else{
                    batch_size = 64;
                    chnl_batch_size = rp_pack_size * resnet_oc / 32768;
                    split_chnl_size = 32768 / rp_pack_size;
                }

                std::cout << resnet_input_size << " " << resnet_output_size << " " << resnet_ic << " " << resnet_oc << " " << resnet_stride << std::endl;
                std::cout << rp_pack_size << " " << batch_size << " " << chnl_batch_size << " " << split_chnl_size << std::endl;

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
                    SPLIT(input_cipher, input_cipher_2, resnet_input_size, resnet_input_size, resnet_oc, rp_pack_size, b * 512, "A", is_gpu, context, pack, enc, eval, edcoder);

                    // IRA2B(input_cipher_2, vgg_input_size, vgg_input_size, rp_pack_size, split_chnl_size, is_gpu, context, pack, enc, eval, edcoder);
                    // print_ct(input_cipher_2, vgg_input_size, vgg_input_size, rp_pack_size, vgg_oc_1, "A", dec, sk);

                    resnet_cipher(input_cipher_2, output_cipher_2, kernel_b1, kernel_b2, kernel_b3, bias_b1, bias_b2, bias_b3, 
                            ss, u, v, ds_bn, resnet_input_size, resnet_stride, rp_pack_size, chnl_batch_size, split_chnl_size, is_gpu, context, pack, enc, dec, sk, eval, edcoder);

                    // MERGE (many->1)
                    MERGE(output_cipher_2, output_cipher, resnet_output_size, resnet_output_size, resnet_oc, rp_pack_size, b * 512, "A", is_gpu, context, pack, enc, eval, edcoder);
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
                print_ct(input_cipher, resnet_output_size, resnet_output_size, rp_pack_size, resnet_oc, "A", dec, sk);
            }
        }

        // AVGPOOL + FC
        std::vector<std::vector<double>> fc_weight(10, std::vector<double>(512, 0));
        std::vector<double> fc_bias(10, 0.0);
        std::vector<HEaaN::Message> fc_output(10, HEaaN::Message(log_slots));
        // Read from pre-trained model
        file.getDataSet("module.fc.weight").read(fc_weight);
        file.getDataSet("module.fc.bias").read(fc_bias);
        // generate_1D_data_double(fc_bias);

        avgpool_fc_plain_2(input, fc_output, fc_weight, fc_bias, context, eval);

        output_cipher.resize(10, HEaaN::Ciphertext(context));
        output_cipher.shrink_to_fit();
        avgpool_fc_cipher_2(input_cipher, output_cipher, fc_weight, fc_bias, context, pack, enc, eval, edcoder, dec, sk);
        std::vector<HEaaN::Message> output_msg(10, HEaaN::Message(log_slots));
        for(int i = 0; i < 10; i++) {
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
            for(int i = 1; i < 10; i++) {
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
            for(int i = 1; i < 10; i++) {
                std::cout << output_msg[i][b * 512].real() << " ";
                if(output_msg[i][b * 512].real() > maxi_ct) {
                    ans_ct = i;
                    maxi_ct = output_msg[i][b * 512].real();
                }
            }
            std::cout << "ANS > " << ans_pt << " " << ans_ct << " " << label[batch_point + b] << std::endl;
            if(ans_pt == label[batch_point + b]) correct_pt++;
            if(ans_ct == label[batch_point + b]) correct_ct++;
        }
        std::cout << "==================================" << std::endl; 
        std::cout << "TOTAL CORRECT PT: " << correct_pt << " CT: " << correct_ct << std::endl; 
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
                   int rp_pack_size, 
                   int chnl_batch_size, int split_chnl_size,
                   bool is_gpu,
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
    int num_chnl_batch = resnet_oc / chnl_batch_size;
    std::vector<double> act(3);

    HEaaN::Message zero(log_slots);
    fillZero(zero);

    std::cout << input_size << " " << output_size << " " << resnet_ic << " " << resnet_oc << std::endl;
    std::cout << "RP PACK SIZE " << rp_pack_size << std::endl;
    std::cout << "number of chnl batch: " << num_chnl_batch << std::endl;

    std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b1_b(num_chnl_batch, 
            std::vector<std::vector<std::vector<double>>>(resnet_ic, 
                std::vector<std::vector<double>>(3, 
                    std::vector<double>(3, 0.0))));
    std::vector<double> bias_b1_b(num_chnl_batch, 0.0);
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b2_b(resnet_oc, 
            std::vector<std::vector<std::vector<double>>>(num_chnl_batch, 
                std::vector<std::vector<double>>(3, 
                    std::vector<double>(3, 0.0))));
    std::vector<double> bias_b2_b(resnet_oc, 0.0);
    std::vector<std::vector<std::vector<std::vector<double>>>> kernel_b3_b(num_chnl_batch, 
            std::vector<std::vector<std::vector<double>>>(resnet_ic, 
                std::vector<std::vector<double>>(1, 
                    std::vector<double>(1, 0.0))));
    std::vector<double> bias_b3_b(num_chnl_batch, 0.0);

    std::vector<HEaaN::Ciphertext> agg_cipher(output_size * output_size, HEaaN::Ciphertext(context));
#pragma omp parallel for
    for(int i = 0; i < output_size * output_size; i++) {
        enc.encrypt(zero, pack, agg_cipher[i]);
        eval.levelDown(agg_cipher[i], 4, agg_cipher[i]);
    }
    
    if(stride == 1) {
        std::cout << "SHORTCUT\n";
        cal_act(act, u[1], v[1], ss[2], ss[3]);
        HEaaN::Plaintext act_0_plain(context);
        HEaaN::Message act_0_msg(log_slots);
        fillRandomComplexKernel(act_0_msg, sqrt(act[0]));
        act_0_plain = edcoder.encode(act_0_msg);
        act_0_plain.setLevel(agg_cipher[0].getLevel() + 1);
#pragma omp parallel for
        for(int i = 0; i < input_size * input_size; i++) {
            HEaaN::Ciphertext tmp_cipher(context);
            tmp_cipher = vec_cipher_1[i];
            eval.levelDown(tmp_cipher, agg_cipher[0].getLevel() + 1, tmp_cipher);
            eval.mult(tmp_cipher, act_0_plain, tmp_cipher);
            eval.add(agg_cipher[i], tmp_cipher, agg_cipher[i]);
        }
    }


    IRA2B(vec_cipher_1, input_size, input_size, rp_pack_size, split_chnl_size, is_gpu, context, pack, enc, eval, edcoder);
    // print_ct(vec_cipher_1, input_size, input_size, rp_pack_size, resnet_oc, "B", dec, sk);

    std::vector<HEaaN::Ciphertext> vec_cipher_3;

    for(int kocb = 0; kocb < chnl_batch_size; kocb++) {
        std::cout << "================= " << std::endl;
        std::cout << "kernel_oc_batch " << kocb << std::endl;
        std::cout << "================= " << std::endl;

        // slice kernel
        for(int oc = 0; oc < num_chnl_batch; oc++) {
            for(int ic = 0; ic < resnet_ic; ic++) {
                for(int i = 0; i < 3; i++) {
                    for(int j = 0; j < 3; j++) {
                        kernel_b1_b[oc][ic][i][j] = kernel_b1[kocb * num_chnl_batch + oc][ic][i][j];
                    }
                }
            }
            bias_b1_b[oc] = bias_b1[kocb * num_chnl_batch + oc];
        }
        for(int oc = 0; oc < resnet_oc; oc++) {
            for(int ic = 0; ic < num_chnl_batch; ic++) {
                for(int i = 0; i < 3; i++) {
                    for(int j = 0; j < 3; j++) {
                        kernel_b2_b[oc][ic][i][j] = kernel_b2[oc][kocb * num_chnl_batch + ic][i][j];
                    }
                }
            }
            bias_b2_b[oc] = bias_b2[oc] / chnl_batch_size;
        }
        for(int oc = 0; oc < num_chnl_batch; oc++) {
            for(int ic = 0; ic < resnet_ic; ic++) {
                for(int i = 0; i < 1; i++) {
                    for(int j = 0; j < 1; j++) {
                        kernel_b3_b[oc][ic][i][j] = kernel_b3[kocb * num_chnl_batch + oc][ic][i][j];
                    }
                }
            }
            bias_b3_b[oc] = bias_b3[kocb * num_chnl_batch + oc];
        }

        std::cout << "FIRST CONV\n";
        vec_cipher_3.resize(conv1_size * conv1_size, HEaaN::Ciphertext(context));
        vec_cipher_3.shrink_to_fit();

        cal_act(act, u[0], v[0], ss[0], ss[1]);
        update_model(kernel_b1_b, bias_b1_b, act);
        padding_cipher(vec_cipher_3, vec_cipher_1, input_size, stride, context, pack, enc, eval);

        if(stride == 1) {
            FCA(vec_cipher_3, vec_cipher_2, kernel_b1_b, bias_b1_b, input_size + 2, input_size + 2, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);
        }else{
            FCA_s2_3_3(vec_cipher_3, kernel_b1_b, bias_b1_b, vec_cipher_2, input_size + 1, input_size + 1, rp_pack_size, "B", is_gpu, context, pack, enc, eval, edcoder);
            // DCA(vec_cipher_3, vec_cipher_2, kernel_b1_b, bias_b1_b, input_size + 1, input_size + 1, 2, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);
        }
        IRB2A(vec_cipher_2, output_size, output_size, rp_pack_size, num_chnl_batch, resnet_oc, is_gpu, context, pack, enc, eval, edcoder);
        act_cipher(vec_cipher_2, act, output_size, output_size, context, edcoder, eval);
        print_ct(vec_cipher_2, output_size, output_size, rp_pack_size, num_chnl_batch, "A", dec, sk);

        // 3*3
        std::cout << "SECOND CONV\n";
        vec_cipher_3.resize(conv2_size * conv2_size, HEaaN::Ciphertext(context));
        vec_cipher_3.shrink_to_fit();

        cal_act(act, u[1], v[1], ss[2], ss[3]);
        update_model(kernel_b2_b, bias_b2_b, act);
        padding_cipher(vec_cipher_3, vec_cipher_2, output_size, 1, context, pack, enc, eval);

        FCA(vec_cipher_3, vec_cipher_2, kernel_b2_b, bias_b2_b, conv2_size, conv2_size, rp_pack_size, true, "A", is_gpu, context, pack, enc, eval, edcoder);

        // shortcut
        if(stride == 2) {
            std::vector<HEaaN::Ciphertext> vec_cipher_4;
            for(int i = 0; i < input_size * input_size; i++) {
                vec_cipher_4.push_back(HEaaN::Ciphertext(context));
                vec_cipher_4[i] = vec_cipher_1[i];
                // eval.levelDown(vec_cipher_4[i], level - 1, vec_cipher_4[i]);
            }

            // 1*1
            std::cout << "THIRD CONV\n";
            vec_cipher_3.resize(output_size * output_size, HEaaN::Ciphertext(context));
            vec_cipher_3.shrink_to_fit();

            cal_act(act, u[1], v[1], ss[2], ss[3]);
            update_model_bn(kernel_b3_b, bias_b3_b, act, ds_bn);

            DCA(vec_cipher_4, vec_cipher_3, kernel_b3_b, bias_b3_b, input_size, input_size, stride, rp_pack_size, true, "B", is_gpu, context, pack, enc, eval, edcoder);

            IRB2A(vec_cipher_3, output_size, output_size, rp_pack_size, num_chnl_batch, resnet_oc, is_gpu, context, pack, enc, eval, edcoder);
            A2B(vec_cipher_3, output_size, output_size, num_chnl_batch, rp_pack_size, is_gpu, context, pack, enc, eval, edcoder);

            // DEFINE FILTER
            HEaaN::Message filter_msg(log_slots);
            fillZero(filter_msg);
            int batch_pos = 0;
            while(batch_pos < 32768) {
                for(int oc = 0; oc < num_chnl_batch; oc++) {
                    filter_msg[batch_pos + oc].real(1);
                }
                batch_pos += rp_pack_size * num_chnl_batch;
            }
            HEaaN::Plaintext filter_plain(context);
            filter_plain = edcoder.encode(filter_msg);
            filter_plain.setLevel(vec_cipher_3[0].getLevel());

#pragma omp parallel for
            for(int i = 0; i < output_size * output_size; i++) {
                eval.mult(vec_cipher_3[i], filter_plain, vec_cipher_3[i]);
                eval.rightRotate(vec_cipher_3[i], num_chnl_batch * kocb, vec_cipher_3[i]);
            }

#pragma omp parallel for
            for(int i = 0; i < output_size * output_size; i++) {
                eval.add(vec_cipher_2[i], vec_cipher_3[i], vec_cipher_2[i]);
            }

            vec_cipher_4.clear();
            vec_cipher_4.shrink_to_fit();
        }

#pragma omp parallel for
        for(int i = 0; i < output_size * output_size; i++) {
            eval.add(agg_cipher[i], vec_cipher_2[i], agg_cipher[i]);
        }
    }

    // distribute
#pragma omp parallel for
    for(int i = 0; i < output_size * output_size; i++) {
        vec_cipher_2[i] = agg_cipher[i];
    }

    // release memory
    agg_cipher.clear();
    agg_cipher.shrink_to_fit();
    vec_cipher_3.clear();
    vec_cipher_3.shrink_to_fit();
}
