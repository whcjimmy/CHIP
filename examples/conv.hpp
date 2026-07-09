#include <HEaaN/Ciphertext.hpp>
#include <HEaaN/KeyPack.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <unistd.h>
#include "utils.cc"
#include "omp.h"

// #include "HEaaN/heaan.hpp"
#include "HEaaN/HEaaN.hpp"

void FCA_B_1_2(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, 
                       HEaaN::Context &context, HEaaN::HomEvaluator &eval);
void FCA_B_2_2(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, 
                       HEaaN::Context &context, HEaaN::HomEvaluator &eval);
void FCA_B_3_3(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, 
                       HEaaN::Context &context, HEaaN::HomEvaluator &eval);
void FCA_A_1_2(int &x, int &y, int &output_row, int &output_col,
                       std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
                                      HEaaN::Context &context, HEaaN::HomEvaluator &eval);
void FCA_A_2_2(int &x, int &y, int &output_row, int &output_col,
                       std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
                                      HEaaN::Context &context, HEaaN::HomEvaluator &eval);
void FCA_A_3_3(int &x, int &y, int &output_row, int &output_col,
                       std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
                                      HEaaN::Context &context, HEaaN::HomEvaluator &eval);

void read_FCA_G_1D(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
                   std::vector<HEaaN::Plaintext> &G_plain, std::string type, int level, int rp_pack_size, 
                   HEaaN::Context &context, HEaaN::EnDecoder &edcoder) {
    int output_pile_row = 1;
    int output_pile_col = 2;
    int kernel_oc = kernel.size();
    int kernel_ic = kernel[0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int icp = pow(2, ceil(log2(kernel_ic)));
    int alpha_row = output_pile_row + kernel_row - 1;
    int alpha_col = output_pile_col + kernel_col - 1;

    const auto log_slots = getLogFullSlots(context);

    // Read G
    std::vector<std::vector<double>>G_a = read_winograd_matrix<double>("G", output_pile_col, kernel_col);

    std::vector<HEaaN::Message> slots_mat;
    for(int i = 0; i < alpha_row; i++) {
        for(int j = 0; j < alpha_col; j++) {
            slots_mat.push_back(HEaaN::Message(log_slots));
            fillZero(slots_mat[i * alpha_col + j]);
        }
    }
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            for(int ic = 0; ic < kernel_ic; ic++) {
                std::vector<std::vector<double>>kernel_T(kernel[oc][ic][0].size(), 
                        std::vector<double>(kernel[oc][ic].size()));
                transpose_matrix(kernel[oc][ic], kernel_T);
                std::vector<std::vector<double>> Gg = matmul(G_a, kernel_T);

                int pos = (type == "A") ? ic * kernel_oc + oc : oc * rp_pack_size + ic;
                for(int i = 0; i < alpha_row; i++) {
                    for(int j = 0; j < alpha_col; j++) {
                        slots_mat[i * alpha_col + j][batch_pos + pos].real(Gg[j][i]);
                        slots_mat[i * alpha_col + j][batch_pos + pos].imag(0);
                    }
                }
            }
        }
        if(type == "A") {
            batch_pos += icp * kernel_oc;
        }else{
            batch_pos += rp_pack_size * kernel_oc;
        }
    }

#pragma omp parallel for
    for(int i = 0; i < alpha_row * alpha_col; i++) {
        G_plain[i] = edcoder.encode(slots_mat[i]);
        G_plain[i].setLevel(level);
    }
}

void read_FCA_G_2D(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
                   std::vector<HEaaN::Plaintext> &G_plain, std::string type, int level, int rp_pack_size,
                   HEaaN::Context &context, HEaaN::EnDecoder &edcoder) {
    int output_pile_row = 2;
    int output_pile_col = 2;
    int kernel_oc = kernel.size();
    int kernel_ic = kernel[0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int icp = pow(2, ceil(log2(kernel_ic)));
    int alpha_row = output_pile_row + kernel_row - 1;
    int alpha_col = output_pile_col + kernel_col - 1;

    const auto log_slots = getLogFullSlots(context);

    // Read G
    std::vector<std::vector<double>>G_a = read_winograd_matrix<double>("G", output_pile_row, kernel_row);
    std::vector<std::vector<double>>G_b = read_winograd_matrix<double>("G", output_pile_col, kernel_col);

    std::vector<HEaaN::Message> slots_mat;
    for(int i = 0; i < alpha_row; i++) {
        for(int j = 0; j < alpha_col; j++) {
            slots_mat.push_back(HEaaN::Message(log_slots));
            fillZero(slots_mat[i * alpha_col + j]);
        }
    }
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            for(int ic = 0; ic < kernel_ic; ic++) {
                std::vector<std::vector<double>> Gg = matmul(G_a, kernel[oc][ic]);
                std::vector<std::vector<double>>G_b_T(G_b[0].size(), std::vector<double>(G_b.size()));
                transpose_matrix(G_b, G_b_T);
                std::vector<std::vector<double>> GgGT = matmul(Gg, G_b_T);

                int pos = (type == "A") ? ic * kernel_oc + oc : oc * rp_pack_size + ic;
                for(int i = 0; i < alpha_row; i++) {
                    for(int j = 0; j < alpha_col; j++) {
                        slots_mat[i * alpha_col + j][batch_pos + pos].real(GgGT[i][j]);
                        slots_mat[i * alpha_col + j][batch_pos + pos].imag(0);
                    }
                }
            }
        }
        if(type == "A") {
            batch_pos += icp * kernel_oc;
        }else{
            batch_pos += rp_pack_size * kernel_oc;
        }
    }

#pragma omp parallel for
    for(int i = 0; i < alpha_row * alpha_col; i++) {
        G_plain[i] = edcoder.encode(slots_mat[i]);
        G_plain[i].setLevel(level);
    }
}

void conv_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
                std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
                std::vector<double> &bias, int stride, HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    int input_row = input[0].size();
    int input_col = input[0][0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int kernel_ic  = kernel[0].size();
    int kernel_oc = kernel.size();
    int output_row = output[0].size();
    int output_col = output[0][0].size();
    const auto log_slots = getLogFullSlots(context);

    std::cout << input_row << " " << input_col << " " << output_row << " " << output_col << std::endl;

    generate_3D_zero(output);

#pragma omp parallel for collapse(4)
    for(int oc = 0; oc < kernel_oc; oc++) {
        for(int ic = 0; ic < kernel_ic; ic++) {
            for(int i = 0; i < output_row; i++) {
                for(int j = 0; j < output_col; j++) {
                    HEaaN::Message tmp(log_slots), sum(log_slots);
                    fillZero(tmp);
                    fillZero(sum);
                    for(int ii = 0; ii < kernel_row; ii++) {
                        for(int jj = 0; jj < kernel_col; jj++) {
                            HEaaN::Message kernel_msg(log_slots);
                            fillRandomComplexKernel(kernel_msg, kernel[oc][ic][ii][jj]);
                            eval.mult(input[ic][i * stride + ii][j * stride + jj], kernel_msg, tmp);
                            eval.add(sum, tmp, sum);
                        }
                    }
                    eval.add(output[oc][i][j], sum, output[oc][i][j]);
                }
            }
        }
    }
#pragma omp parallel for collapse(3)
    for(int oc = 0; oc < kernel_oc; oc++) {
        for(int i = 0; i < output_row; i++) {
            for(int j = 0; j < output_col; j++) {
                HEaaN::Message bias_msg(log_slots);
                fillRandomComplexKernel(bias_msg, bias[oc]);
                eval.add(output[oc][i][j], bias_msg, output[oc][i][j]);
            }
        }
    }

    print_pt(output);
}

void DCA(std::vector<HEaaN::Ciphertext> &input_cipher, 
         std::vector<HEaaN::Ciphertext> &output_cipher,
         std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
         std::vector<double> &bias,
         int input_row, int input_col, int stride, int rp_pack_size, bool is_bias, 
         std::string type, bool is_gpu,
         HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
         HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int kernel_oc = kernel.size();
    int kernel_ic = kernel[0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int icp = pow(2, ceil(log2(kernel_ic)));
    int output_row = int(ceil(1.0 * (input_row - kernel_row + 1) / stride));
    int output_col = int(ceil(1.0 * (input_col - kernel_col + 1) / stride));

    // ENCRYPT OUTPUT
    HEaaN::Message zero(log_slots);
    fillZero(zero);
#pragma omp parallel for
    for(int i = 0; i < output_row * output_col; i++) {
        enc.encrypt(zero, pack, output_cipher[i]);
        eval.levelDown(output_cipher[i], input_cipher[0].getLevel() - 1, output_cipher[i]);
    }

    // Encode Kernel
    std::vector<HEaaN::Message> kernel_slots(kernel_row * kernel_col, HEaaN::Message(log_slots));
#pragma omp parallel for 
    for(int i = 0; i < kernel_row * kernel_col; i++) {
        fillZero(kernel_slots[i]);
    }
    int batch_pos = 0;
    while (batch_pos < 32768) {
#pragma omp parallel for collapse(4)
        for(int oc = 0; oc < kernel_oc; oc++) {
            for(int ic = 0; ic < kernel_ic; ic++) {
                for(int i = 0; i < kernel_row; i++) {
                    for(int j = 0; j < kernel_col; j++) {
                        int pos = i * kernel_col + j;
                        int slots_pos = (type == "A") ? ic * kernel_oc + oc : oc * rp_pack_size + ic;
                        kernel_slots[pos][batch_pos + slots_pos].real(kernel[oc][ic][i][j]);
                        kernel_slots[pos][batch_pos + slots_pos].imag(0);
                    }
                }
            }
        }
        batch_pos += rp_pack_size * kernel_oc;
    }
    std::vector<HEaaN::Plaintext> kernel_plain(kernel_row * kernel_col, HEaaN::Plaintext(context));

#pragma omp parallel for
    for(int i = 0; i < kernel_row * kernel_col; i++) {
        kernel_plain[i] = edcoder.encode(kernel_slots[i]);
        kernel_plain[i].setLevel(input_cipher[0].getLevel());
    }

    // Encode Bias
    HEaaN::Plaintext bias_plain(context);
    if(is_bias) {
        HEaaN::Message bias_msg(log_slots);
        fillZero(bias_msg);

        int batch_pos = 0;
        while(batch_pos < 32768) {
            for(int oc = 0; oc < kernel_oc; oc++) {
                int pos = (type == "A") ?  pos = oc : pos = rp_pack_size * oc;
                bias_msg[batch_pos + pos].real(bias[oc]);
            }
            batch_pos += rp_pack_size * kernel_oc;
        }

        bias_plain = edcoder.encode(bias_msg);
        bias_plain.setLevel(input_cipher[0].getLevel() - 1);
    }

    if(is_gpu) {
        std::cout << "BEGIN LOAD\n";
        // LOAD input, output, krenel, and bias to GPU
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        for(int i = 0; i < output_row * output_col; i++) {
            output_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        for(int i = 0; i < kernel_row * kernel_col; i++) {
            kernel_plain[i].to(HEaaN::DeviceType::GPU);
        }
        bias_plain.to(HEaaN::DeviceType::GPU);

        std::cout << "Load all public keys to GPU memory ..." << std::endl;
        pack.to(HEaaN::DeviceType::GPU);
        std::cout << "END LOAD\n";
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("DCA ");

#pragma omp parallel for collapse(2) if(!is_gpu)
    for(int i = 0; i < output_row; i++) {
        for(int j = 0; j < output_col; j++) {
            HEaaN::Ciphertext sum_cipher(context), tmp_cipher(context);
            for(int ii = 0; ii < kernel_row; ii++) {
                for(int jj = 0; jj < kernel_col; jj++) {
                    int input_pos = (i * stride + ii) * input_col + (j * stride + jj);
                    eval.multWithoutRescale(input_cipher[input_pos], kernel_plain[ii * kernel_col + jj], tmp_cipher);
                    if(ii * kernel_col + jj == 0) {
                        sum_cipher = tmp_cipher;
                    }else{
                        eval.add(sum_cipher, tmp_cipher, sum_cipher);
                    }
                }
            }
            output_cipher[i * output_col + j] = sum_cipher;
        }
    }

#pragma omp parallel for
    for(int i = 0; i < output_row * output_col; i++) {
        HEaaN::Ciphertext tmp_cipher(context);
        eval.rescale(output_cipher[i]); // lazy rescaling
        tmp_cipher = output_cipher[i];
        for(int p = log2(icp) - 1; p >= 0; p--) {
            int rot_pos = (type == "A") ? pow(2, p) * kernel_oc : pow(2, p);
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(output_cipher[i], tmp_cipher, output_cipher[i]);
            tmp_cipher = output_cipher[i];
        }
        if(is_bias) {
            eval.add(output_cipher[i], bias_plain, output_cipher[i]);
        }
    }

    timer.end();
    timer.print();

    if(is_gpu) {
        // LOAD input_cipher and output_cipher to CPU
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        for(int i = 0; i < output_row * output_col; i++) {
            output_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        for(int i = 0; i < kernel_row * kernel_col; i++) {
            kernel_plain[i].to(HEaaN::DeviceType::CPU);
        }
        bias_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }

    kernel_slots.clear();
    kernel_slots.shrink_to_fit();
    kernel_plain.clear();
    kernel_plain.shrink_to_fit();
}

void FCA(std::vector<HEaaN::Ciphertext> &input_cipher, 
         std::vector<HEaaN::Ciphertext> &output_cipher,
         std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
         std::vector<double> &bias,
         int input_row, int input_col, int rp_pack_size, bool is_bias,
         std::string type, bool is_gpu,
         HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
         HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int kernel_oc = kernel.size();
    int kernel_ic = kernel[0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int icp = pow(2, ceil(log2(kernel_ic)));

    int output_pile_row = kernel_row == 1 ? 1 : 2;
    int output_pile_col = 2;
    int alpha_row = kernel_row + output_pile_row - 1;
    int alpha_col = kernel_col + output_pile_col - 1;
    int overlap_row = kernel_row - 1;
    int overlap_col = kernel_col - 1;
    int num_output_row = int((input_row - overlap_row) / (alpha_row - overlap_row));
    int num_output_col = int((input_col - overlap_col) / (alpha_col - overlap_col));
    int output_row = num_output_row * output_pile_row;
    int output_col = num_output_col * output_pile_col;

    // ENCRYPT OUTPUT
    HEaaN::Message zero(log_slots);
    fillZero(zero);
#pragma omp parallel for
    for(int i = 0; i < output_row * output_col; i++) {
        enc.encrypt(zero, pack, output_cipher[i]);
        eval.levelDown(output_cipher[i], input_cipher[0].getLevel() - 1, output_cipher[i]);
    }

    std::cout << "input_cipher level " << input_cipher[0].getLevel() << std::endl;
    std::cout << "output_cipher level " << output_cipher[0].getLevel() << std::endl;

    // Encode Kernel
    std::vector<HEaaN::Plaintext> G_plain(alpha_row * alpha_col, HEaaN::Plaintext(context));
    if(kernel_row == 1) {
        read_FCA_G_1D(kernel, G_plain, type, input_cipher[0].getLevel(), rp_pack_size, context, edcoder);
    }else {
        read_FCA_G_2D(kernel, G_plain, type, input_cipher[0].getLevel(), rp_pack_size, context, edcoder);
    }

    // Encode Bias
    HEaaN::Plaintext bias_plain(context);
    if(is_bias) {
        HEaaN::Message bias_msg(log_slots);
        fillZero(bias_msg);

        int batch_pos = 0;
        while(batch_pos < 32768) {
            for(int oc = 0; oc < kernel_oc; oc++) {
                int pos = (type == "A") ?  oc : rp_pack_size * oc;
                bias_msg[batch_pos + pos].real(bias[oc]);
            }
            batch_pos += rp_pack_size * kernel_oc;
        }

        bias_plain = edcoder.encode(bias_msg);
        bias_plain.setLevel(input_cipher[0].getLevel() - 1);
    }

    /*
    std::vector<std::vector<HEaaN::Ciphertext>> d(num_output_row * num_output_col, std::vector<HEaaN::Ciphertext>(alpha_row * alpha_col, HEaaN::Ciphertext(context)));
    std::vector<std::vector<HEaaN::Ciphertext>> D(num_output_row * num_output_col, std::vector<HEaaN::Ciphertext>(alpha_row * alpha_col, HEaaN::Ciphertext(context)));

#pragma omp parallel for collapse(2)
    for(int i = 0; i < num_output_row; i++) {
        for(int j = 0; j < num_output_col; j++) {
            int tile_pos = i * num_output_col + j;
            int x = output_pile_row * i;
            int y = output_pile_col * j;

            for(int ii = 0; ii < alpha_row; ii++) {
                for(int jj = 0; jj < alpha_col; jj++) {
                    int pos = ii * alpha_col + jj;
                    int input_pos = (x + ii) * input_col + (y + jj);
                    d[tile_pos][pos] = input_cipher[input_pos];
                    D[tile_pos][pos] = input_cipher[input_pos]; 
                }
            }
        }
    }
    */

    if(is_gpu) {
        std::cout << "BEGIN LOAD\n";
        // LOAD G_plain to GPU
        for(int i = 0; i < alpha_row * alpha_col; i++) { 
            G_plain[i].to(HEaaN::DeviceType::GPU);
        }
        // LOAD bias to GPU
        bias_plain.to(HEaaN::DeviceType::GPU);
        // LOAD output_cipher to GPU
        for(int i = 0; i < output_row * output_col; i++) {
            output_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        std::cout << "Load all public keys to GPU memory ..." << std::endl;
        pack.to(HEaaN::DeviceType::GPU);
        std::cout << "END LOAD\n";
    }

    process_mem_usage(vm2, rss2);
    std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;

    HEaaN::HEaaNTimer timer(is_gpu);
    if(!is_gpu) timer.start("FCA ");

#pragma omp parallel for collapse(2) if(!is_gpu)
    for(int i = 0; i < num_output_row; i++) {
        for(int j = 0; j < num_output_col; j++) {
            int tile_pos = i * num_output_col + j;
            int x = output_pile_row * i;
            int y = output_pile_col * j;

            std::vector<HEaaN::Ciphertext> d;
            std::vector<HEaaN::Ciphertext> D;

#pragma omp parallel for collapse(2) if(!is_gpu)
            for(int ii = 0; ii < alpha_row; ii++) {
                for(int jj = 0; jj < alpha_col; jj++) {
                    int pos = ii * alpha_col + jj;
                    int input_pos = (x + ii) * input_col + (y + jj);
                    d.push_back(input_cipher[input_pos]);
                    D.push_back(input_cipher[input_pos]);
                }
            }

            if(is_gpu) {
                // LOAD d and D to GPU
                for(int ii = 0; ii < alpha_row * alpha_col; ii++) {
                    d[ii].to(HEaaN::DeviceType::GPU);
                    D[ii].to(HEaaN::DeviceType::GPU);
                }
                timer.start("FCA ");
            }

            if(kernel_row == 1) {
                FCA_B_1_2(d, D, context, eval);
            }else{
                if(kernel_row == 2 && kernel_col == 2) {
                    FCA_B_2_2(d, D, context, eval);
                }else if(kernel_row == 3 && kernel_col == 3) {
                    FCA_B_3_3(d, D, context, eval);
                }else if(kernel_row == 2 && kernel_col == 3) {
                    // FCA_B_2_2_2_3(d, D);
                }
            }

#pragma omp parallel for
            for(int ii = 0; ii < alpha_row * alpha_col; ii++) {
                eval.multWithoutRescale(D[ii], G_plain[ii], D[ii]); // lazy rescaling
            }

            if(kernel_row == 1) {
                FCA_A_1_2(x, y, output_row, output_col, D, output_cipher, context, eval);
            }else{
                if(kernel_row == 2 && kernel_col == 2) {
                    FCA_A_2_2(x, y, output_row, output_col, D, output_cipher, context, eval);
                }else if(kernel_row == 3 && kernel_col == 3){
                    FCA_A_3_3(x, y, output_row, output_col, D, output_cipher, context, eval);
                }else if(kernel_row == 2 && kernel_col == 3){
                    // FCA_A_2_2_2_3(x, y, 0, output_row, output_col, D, out);
                }
            }
            
            if(is_gpu) {
                timer.end();
                // LOAD d and D to GPU
                for(int i = 0; i < alpha_row * alpha_col; i++) {
                    d[i].to(HEaaN::DeviceType::CPU);
                    D[i].to(HEaaN::DeviceType::CPU);
                }
            }

            d.clear();
            d.shrink_to_fit();
            D.clear();
            D.shrink_to_fit();
        }
    }

    process_mem_usage(vm2, rss2);
    std::cout << "CURRENT RAM USAGE " << (vm2)/(1024 * 1024) << " GB" << std::endl;

    if(is_gpu) timer.start("FCA ");

#pragma omp parallel for if(!is_gpu)
    for(int i = 0; i < output_row * output_col; i++) {
        HEaaN::Ciphertext tmp_cipher(context);
        eval.rescale(output_cipher[i]); // lazy rescaling
        tmp_cipher = output_cipher[i];
        for(int p = log2(icp) - 1; p >= 0; p--) {
            int rot_pos = (type == "A") ? pow(2, p) * kernel_oc : pow(2, p);
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(output_cipher[i], tmp_cipher, output_cipher[i]);
            tmp_cipher = output_cipher[i];
        }
        if(is_bias) {
            eval.add(output_cipher[i], bias_plain, output_cipher[i]);
        }
    }

    timer.end();
    timer.print();

    if(is_gpu) {
        // LOAD G_plain, bias, and output back to CPU
        for(int i = 0; i < alpha_row * alpha_col; i++) {
            G_plain[i].to(HEaaN::DeviceType::CPU);
        }
        bias_plain.to(HEaaN::DeviceType::CPU);
        for(int i = 0; i < output_row * output_col; i++) {
            output_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        pack.to(HEaaN::DeviceType::CPU);
    }

    // d.clear();
    // d.shrink_to_fit();
    // D.clear();
    // D.shrink_to_fit();
    G_plain.clear();
    G_plain.shrink_to_fit();
}

void FCA_s2_3_3(
        std::vector<HEaaN::Ciphertext> &input_cipher,
        std::vector<std::vector<std::vector<std::vector<double>>>> &kernel,
        std::vector<double> &bias,
        std::vector<HEaaN::Ciphertext> &output_cipher,
        int input_row, int input_col, int rp_pack_size, 
        std::string type, bool is_gpu,
        HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
        HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int kernel_oc  = kernel.size();
    int kernel_ic  = kernel[0].size();
    int kernel_row = kernel[0][0].size();
    int kernel_col = kernel[0][0][0].size();
    int icp = pow(2, ceil(log2(kernel_ic)));

    int output_pile_row = 2;
    int output_pile_col = 2;
    int alpha_row = kernel_row + output_pile_row - 1;
    int alpha_col = kernel_col + output_pile_col - 1;
    int overlap_row = kernel_row - 1;
    int overlap_col = kernel_col - 1;

    // SPLIT KERNEL
    std::vector<std::vector<std::vector<std::vector<double>>>> 
        KA(kernel_oc, std::vector<std::vector<std::vector<double>>>(
           kernel_ic, std::vector<std::vector<double>>(2, std::vector<double>(2))));
    std::vector<std::vector<std::vector<std::vector<double>>>> 
        KB(kernel_oc, std::vector<std::vector<std::vector<double>>>(
           kernel_ic, std::vector<std::vector<double>>(1, std::vector<double>(2))));
    std::vector<std::vector<std::vector<std::vector<double>>>> 
        KC(kernel_oc, std::vector<std::vector<std::vector<double>>>(
           kernel_ic, std::vector<std::vector<double>>(1, std::vector<double>(2))));
    std::vector<std::vector<std::vector<std::vector<double>>>> 
        KD(kernel_oc, std::vector<std::vector<std::vector<double>>>(
           kernel_ic, vector<vector<double>>(1, vector<double>(1))));
    split_kernel(kernel, KA, KB, KC, KD);

    // SPLIT INPUT MATRIX SIZE
    int input_row_even = ceil(input_row / 2.0);
    int input_row_odd  =  int(input_row / 2.0);
    int input_col_even = ceil(input_col / 2.0);
    int input_col_odd  =  int(input_col / 2.0);

    // SPLIT OUTPUT MATRIX SIZE
    int output_row = int(ceil(1.0 * (input_row_even - 2 + 1)));
    int output_col = int(ceil(1.0 * (input_col_even - 2 + 1)));
    std::cout << output_row << " " << output_col << std::endl;

    // ENCRYPT OUTPUT
    HEaaN::Message zero(log_slots);
    fillZero(zero);
#pragma omp parallel for
    for(int i = 0; i < output_row * output_col; i++) {
        enc.encrypt(zero, pack, output_cipher[i]);
        eval.levelDown(output_cipher[i], input_cipher[0].getLevel() - 1, output_cipher[i]);
    }

    // Input and Output Ciphertext Vector
    std::vector<HEaaN::Ciphertext> I(input_row_even * input_col_even, HEaaN::Ciphertext(context));
    std::vector<HEaaN::Ciphertext> O(output_row * output_col, HEaaN::Ciphertext(context));

    {
        // INPUT MATRIX TYPE A
        int input_row_p = input_row_even;
        int input_col_p = input_row_even;

        split_matrix(input_cipher, I, "A", input_row, input_col, input_row_p, input_col_p);

        FCA(I, O, KA, bias, input_row_p, input_col_p, rp_pack_size, true, type, is_gpu, context, pack, enc, eval, edcoder);

        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::GPU);
                O[i].to(HEaaN::DeviceType::GPU);
            }
            pack.to(HEaaN::DeviceType::GPU);
        }
        // SUM
#pragma omp parallel for
        for(int i = 0; i < output_row * output_col; i++) {
            eval.add(output_cipher[i], O[i], output_cipher[i]);
        }
        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::CPU);
                O[i].to(HEaaN::DeviceType::CPU);
            }
            pack.to(HEaaN::DeviceType::CPU);
        }
    }

    {
        // INPUT MATRIX TYPE B
        int input_row_p = input_row_odd;
        int input_col_p = input_row_even;

        split_matrix(input_cipher, I, "B", input_row, input_col, input_row_p, input_col_p);

        FCA(I, O, KB, bias, input_row_p, input_col_p, rp_pack_size, false, type, is_gpu, context, pack, enc, eval, edcoder);

        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::GPU);
                O[i].to(HEaaN::DeviceType::GPU);
            }
            pack.to(HEaaN::DeviceType::GPU);
        }
        // SUM
#pragma omp parallel for collapse(2)
        for(int i = 0; i < output_row; i++) {
            for(int j = 0; j < output_col; j++) {
                int pos = i * output_col + j;
                int ob_pos = j * output_col + i;
                eval.add(output_cipher[pos], O[ob_pos], output_cipher[pos]);
            }
        }
        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::CPU);
                O[i].to(HEaaN::DeviceType::CPU);
            }
            pack.to(HEaaN::DeviceType::CPU);
        }
    }

    {
        // INPUT MATRIX TYPE C
        int input_row_p = input_row_odd;
        int input_col_p = input_row_even;

        split_matrix(input_cipher, I, "C", input_row, input_col, input_row_p, input_col_p);

        FCA(I, O, KC, bias, input_row_p, input_col_p, rp_pack_size, false, type, is_gpu, context, pack, enc, eval, edcoder);

        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::GPU);
                O[i].to(HEaaN::DeviceType::GPU);
            }
            pack.to(HEaaN::DeviceType::GPU);
        }
        // SUM
#pragma omp parallel for 
        for(int i = 0; i < output_row * output_col; i++) {
            eval.add(output_cipher[i], O[i], output_cipher[i]);
        }
        if(is_gpu) {
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::CPU);
                O[i].to(HEaaN::DeviceType::CPU);
            }
            pack.to(HEaaN::DeviceType::CPU);
        }
    }

    {
        // INPUT MATRIX TYPE D
        int input_row_p = input_row_odd;
        int input_col_p = input_row_odd;

        split_matrix(input_cipher, I, "D", input_row, input_col, input_row_p, input_col_p);

        HEaaN::Message KD_msg(log_slots);
        fillZero(KD_msg);
        int batch_pos = 0;
        while(batch_pos < 32768) {
            for(int oc = 0; oc < kernel_oc; oc++) {
                for(int ic = 0; ic < kernel_ic; ic++) {
                    int pos = (type == "A") ? ic * kernel_oc + oc : oc * rp_pack_size + ic;
                    KD_msg[batch_pos + pos].real(KD[oc][ic][0][0]);
                }
            }
            batch_pos += rp_pack_size * kernel_oc;
        }

        if(is_gpu) {
            for(int i = 0; i < input_row_p * input_col_p; i++) {
                I[i].to(HEaaN::DeviceType::GPU);
            }
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::GPU);
                O[i].to(HEaaN::DeviceType::GPU);
            }
            KD_msg.to(HEaaN::DeviceType::GPU);
            pack.to(HEaaN::DeviceType::GPU);
        }

        HEaaN::HEaaNTimer timer(true);
        timer.start("FCA ");
#pragma omp parallel for collapse(2)
        for(int i = 0; i < output_row; i++) {
            for(int j = 0; j < output_col; j++) {
                int pos = i * output_col + j;
                eval.mult(I[pos], KD_msg, I[pos]);
                HEaaN::Ciphertext tmp_cipher = I[pos];
                for(int p = log2(icp) - 1; p >= 0; p--) {
                    int rot_pos = (type == "A") ? pow(2, p) * kernel_oc : pow(2, p);
                    // std::cout << "ROT LEFT == " << rot_pos << std::endl;
                    eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
                    eval.add(I[pos], tmp_cipher, I[pos]);
                    tmp_cipher = I[pos];
                }
                eval.add(output_cipher[pos], tmp_cipher, output_cipher[pos]);
            }
        }
        timer.end();
        timer.print();

        if(is_gpu) {
            for(int i = 0; i < input_row_p * input_col_p; i++) {
                I[i].to(HEaaN::DeviceType::CPU);
            }
            for(int i = 0; i < output_row * output_col; i++) {
                output_cipher[i].to(HEaaN::DeviceType::CPU);
                O[i].to(HEaaN::DeviceType::CPU);
            }
            KD_msg.to(HEaaN::DeviceType::CPU);
            pack.to(HEaaN::DeviceType::CPU);
        }
    }

    for(int i = 0; i < output_row * output_col; i++) {
        output_cipher[i].to(HEaaN::DeviceType::CPU);
    }
    pack.to(HEaaN::DeviceType::CPU);

    I.clear();
    I.shrink_to_fit();
    O.clear();
    O.shrink_to_fit();
}

void SPLIT(std::vector<HEaaN::Ciphertext> &input_cipher, 
           std::vector<HEaaN::Ciphertext> &input_cipher_2,
           int input_row, int input_col, int kernel_oc, 
           int rp_pack_size, int gap, 
           std::string type, bool is_gpu,
           HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
           HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int level = input_cipher[0].getLevel();

    // DEFINE FILTER
    HEaaN::Message filter_msg(log_slots);
    fillZero(filter_msg);
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            filter_msg[batch_pos + oc + gap].real(1);
        }
        batch_pos += rp_pack_size * kernel_oc;
    }
    HEaaN::Plaintext filter_plain(context);
    filter_plain = edcoder.encode(filter_msg);
    filter_plain.setLevel(level);

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::GPU);
            input_cipher_2[i].to(HEaaN::DeviceType::GPU);
        }
        filter_plain.to(HEaaN::DeviceType::GPU);
        pack.to(HEaaN::DeviceType::GPU);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("SPLIT ");
    std::cout << "PHASE 1\n";
#pragma omp parallel for if(!is_gpu)
    for(int ii = 0; ii < input_row * input_col; ii++) {
        eval.mult(input_cipher[ii], filter_plain, input_cipher_2[ii]);
        // std::cout << "ROT LEFT == " << gap << std::endl;
        eval.leftRotate(input_cipher_2[ii], gap, input_cipher_2[ii]);
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::CPU);
            input_cipher_2[i].to(HEaaN::DeviceType::CPU);
        }
        filter_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }
}

void MERGE(std::vector<HEaaN::Ciphertext> &output_cipher_2, 
           std::vector<HEaaN::Ciphertext> &output_cipher, 
           int input_row, int input_col, int kernel_oc,
           int rp_pack_size, int gap,
           std::string type, bool is_gpu,
           HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
           HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int level = output_cipher_2[0].getLevel();

    // DEFINE FILTER
    HEaaN::Message filter_msg(log_slots);
    fillZero(filter_msg);
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            filter_msg[batch_pos + oc].real(1);
        }
        batch_pos += rp_pack_size * kernel_oc;
    }
    HEaaN::Plaintext filter_plain(context);
    filter_plain = edcoder.encode(filter_msg);
    filter_plain.setLevel(level);

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            output_cipher_2[i].to(HEaaN::DeviceType::GPU);
            output_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        filter_plain.to(HEaaN::DeviceType::GPU);
        pack.to(HEaaN::DeviceType::GPU);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("MERGE ");
    std::cout << "PHASE 1\n";
#pragma omp parallel for
    for(int ii = 0; ii < input_row * input_col; ii++) {
        eval.mult(output_cipher_2[ii], filter_plain, output_cipher_2[ii]);
        // std::cout << "ROT RIGHT == " << gap << std::endl;
        eval.rightRotate(output_cipher_2[ii], gap, output_cipher_2[ii]);
        eval.add(output_cipher_2[ii], output_cipher[ii], output_cipher[ii]);
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            output_cipher_2[i].to(HEaaN::DeviceType::CPU);
            output_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        filter_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }
}

void IRA2B(std::vector<HEaaN::Ciphertext> &input_cipher, 
         int input_row, int input_col, int rp_pack_size, int kernel_oc, bool is_gpu,
         HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
         HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int ocp   = pow(2, ceil(log2(kernel_oc)));
    std::cout << "IRA2B: " << ocp << std::endl;

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        // filter_plain.to(HEaaN::DeviceType::GPU);
        pack.to(HEaaN::DeviceType::GPU);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("IRA2B ");
    std::cout << "PHASE 2\n";
#pragma omp parallel for 
    for(int ii = 0; ii < input_row * input_col; ii++) {
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = input_cipher[ii];
        for(int p = 0; p < log2(ocp); p++) {
            int rot_pos = pow(2, p) * rp_pack_size;
            // std::cout << "ROT RIGHT == " << rot_pos << std::endl;
            eval.rightRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(input_cipher[ii], tmp_cipher, input_cipher[ii]);
            tmp_cipher = input_cipher[ii];
        }
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        // filter_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }
}

void IRB2A(std::vector<HEaaN::Ciphertext> &input_cipher, 
        int input_row, int input_col, int rp_pack_size, int kernel_oc, int kernel_oc_n, 
        bool is_gpu,
        HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
        HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    // int ocp   = pow(2, ceil(log2(kernel_oc)));
    int level = input_cipher[0].getLevel();
    std::cout << "IRB2A " << rp_pack_size << " " << kernel_oc << " " << kernel_oc_n << std::endl;

    // DEFINE FILTER
    HEaaN::Message filter_msg(log_slots);
    fillZero(filter_msg);
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            int filter_pos = oc * rp_pack_size;
            filter_msg[batch_pos + filter_pos].real(1);
        }
        batch_pos += rp_pack_size * kernel_oc;
    }
    HEaaN::Plaintext filter_plain(context);
    filter_plain = edcoder.encode(filter_msg);
    filter_plain.setLevel(level);

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        filter_plain.to(HEaaN::DeviceType::GPU);
        pack.to(HEaaN::DeviceType::GPU);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("IR ");
    std::cout << "PHASE 1\n";
#pragma omp parallel for
    for(int ii = 0; ii < input_row * input_col; ii++) {
        eval.mult(input_cipher[ii], filter_plain, input_cipher[ii]);
    }

    std::cout << "PHASE 2\n";
#pragma omp parallel for 
    for(int ii = 0; ii < input_row * input_col; ii++) {
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = input_cipher[ii];
        for(int p = 0; p < log2(kernel_oc_n); p++) {
            int rot_pos = pow(2, p);
            // std::cout << "ROT RIGHT == " << rot_pos << std::endl;
            eval.rightRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(input_cipher[ii], tmp_cipher, input_cipher[ii]);
            tmp_cipher = input_cipher[ii];
        }
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        filter_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }
}

void A2B(std::vector<HEaaN::Ciphertext> &input_cipher, 
        int input_row, int input_col, int kernel_oc, int rp_pack_size, bool is_gpu,
        HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
        HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder) {
    const auto log_slots = getLogFullSlots(context);

    int ocp   = pow(2, ceil(log2(kernel_oc)));
    int level = input_cipher[0].getLevel();
    std::cout << ocp << std::endl;

    // DEFINE FILTER
    HEaaN::Message filter_msg(log_slots);
    fillZero(filter_msg);
    int batch_pos = 0;
    while(batch_pos < 32768) {
        for(int oc = 0; oc < kernel_oc; oc++) {
            int filter_pos = oc * rp_pack_size + oc;
            filter_msg[batch_pos + filter_pos].real(1);
        }
        batch_pos += rp_pack_size * kernel_oc;
    }
    HEaaN::Plaintext filter_plain(context);
    filter_plain = edcoder.encode(filter_msg);
    filter_plain.setLevel(level);

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        filter_plain.to(HEaaN::DeviceType::GPU);
        pack.to(HEaaN::DeviceType::GPU);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("A2B ");
    std::cout << "PHASE 1\n";
#pragma omp parallel for
    for(int ii = 0; ii < input_row * input_col; ii++) {
        eval.mult(input_cipher[ii], filter_plain, input_cipher[ii]);
    }
    
    std::cout << "PHASE 2\n";
#pragma omp parallel for 
    for(int ii = 0; ii < input_row * input_col; ii++) {
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = input_cipher[ii];
        for(int p = log2(ocp) - 1; p >= 0; p--) {
            int rot_pos = pow(2, p) * rp_pack_size;
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(input_cipher[ii], tmp_cipher, input_cipher[ii]);
            tmp_cipher = input_cipher[ii];
        }
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        for(int i = 0; i < input_row * input_col; i++) {
            input_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        filter_plain.to(HEaaN::DeviceType::CPU);
        pack.to(HEaaN::DeviceType::CPU);
    }
}

void cal_act(std::vector<double> &act, std::vector<double> &u, std::vector<double> &v, double scale, double shift) {
    const double PI  = 3.14159265358979323846;
    const double eps = 0.0001;
    // BUG: v should be v**2
    act[0] = scale / (sqrt(8*PI*(v[1]+eps)));
    act[1] = scale / (2*sqrt(v[0]+eps));
    act[2] = scale * (-u[0]/(2*sqrt(v[0]+eps)) - (1+sqrt(2)*u[1])/(sqrt(8*PI*(v[1]+eps)))) + shift;
    std::cout << act[0] << " " << act[1] << " " << act[2] << std::endl;
}

void act_plain_1d(std::vector<HEaaN::Message> &input, std::vector<double> &act, 
                  HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::cout << "ACT\n";
    const auto log_slots = getLogFullSlots(context);
    int input_size = input.size();

#pragma omp parallel for 
    for(int i = 0; i < input_size; i++) {
        std::vector<HEaaN::Message> act_msg(3, HEaaN::Message(log_slots));
        fillRandomComplexKernel(act_msg[0], act[0]);
        fillRandomComplexKernel(act_msg[1], act[1]);
        fillRandomComplexKernel(act_msg[2], act[2]);
        eval.mult(input[i], act_msg[1], act_msg[1]);
        eval.mult(input[i], input[i],   input[i]);
        eval.mult(input[i], act_msg[0], input[i]);
        eval.add( input[i], act_msg[1], input[i]);
        eval.add( input[i], act_msg[2], input[i]);
    }
    // print_pt(input);
}

void act_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
        std::vector<double> &act, HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::cout << "ACT\n";
    const auto log_slots = getLogFullSlots(context);
    int kernel_oc = input.size();
    int input_size = input[0].size();

#pragma omp parallel for collapse(3)
    for(int oc = 0; oc < kernel_oc; oc++) {
        for(int i = 0; i < input_size; i++) {
            for(int j = 0; j < input_size; j++) {
                std::vector<HEaaN::Message> act_msg(3, HEaaN::Message(log_slots));
                fillRandomComplexKernel(act_msg[0], act[0]);
                fillRandomComplexKernel(act_msg[1], act[1]);
                fillRandomComplexKernel(act_msg[2], act[2]);
                eval.mult(input[oc][i][j], act_msg[1], act_msg[1]);
                eval.mult(input[oc][i][j], input[oc][i][j], input[oc][i][j]);
                eval.mult(input[oc][i][j], act_msg[0], input[oc][i][j]);
                eval.add( input[oc][i][j], act_msg[1], input[oc][i][j]);
                eval.add( input[oc][i][j], act_msg[2], input[oc][i][j]);
            }
        }
    }
    print_pt(input);
}

void act_cipher(std::vector<HEaaN::Ciphertext> &input_cipher, std::vector<double> &act, 
                int row, int col, HEaaN::Context &context, HEaaN::EnDecoder &edcoder, HEaaN::HomEvaluator &eval) {
    std::cout << "ACT\n";

    const auto log_slots = getLogFullSlots(context);
    int level = input_cipher[0].getLevel();
    HEaaN::Plaintext act_1_plain(context), act_2_plain(context);
    HEaaN::Message act_1_msg(log_slots), act_2_msg(log_slots);
    fillRandomComplexKernel(act_1_msg, act[1]);
    fillRandomComplexKernel(act_2_msg, act[2]);
    act_1_plain = edcoder.encode(act_1_msg);
    act_1_plain.setLevel(level);
    act_2_plain = edcoder.encode(act_2_msg);
    act_2_plain.setLevel(level - 1);

    HEaaN::HEaaNTimer timer(true);
    timer.start("ACT ");
#pragma omp parallel for
    for(int i = 0; i < row * col; i++) {
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = input_cipher[i];
        eval.mult(tmp_cipher, act_1_plain, tmp_cipher);
        eval.mult(input_cipher[i], input_cipher[i], input_cipher[i]);
        eval.add(input_cipher[i], tmp_cipher, input_cipher[i]);
        eval.add(input_cipher[i], act_2_plain, input_cipher[i]);
    }
    timer.end();
    timer.print();
}

void update_model(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel, std::vector<double> &bias, std::vector<double> &act) {
    for(int oc = 0; oc < kernel.size(); oc++) {
        for(int ic = 0; ic < kernel[0].size(); ic++) {
            for(int i = 0; i < kernel[0][0].size(); i++) {
                for(int j = 0; j < kernel[0][0][0].size(); j++) {
                    kernel[oc][ic][i][j] *= sqrt(act[0]);
                }
            }
        }
        bias[oc] *= sqrt(act[0]);
    }
    act[1] = act[1] / sqrt(act[0]);
}

void update_model_bn(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel, std::vector<double> &bias, std::vector<double> &act, std::vector<std::vector<double>> &ds_bn) {
    for(int oc = 0; oc < kernel.size(); oc++) {
        for(int ic = 0; ic < kernel[0].size(); ic++) {
            for(int i = 0; i < kernel[0][0].size(); i++) {
                for(int j = 0; j < kernel[0][0][0].size(); j++) {
                    kernel[oc][ic][i][j] *= sqrt(act[0]) * ds_bn[0][oc] / sqrt(ds_bn[3][oc] + 0.00001);
                }
            }
        }
        bias[oc] = bias[oc] - ds_bn[2][oc];
        bias[oc] *= sqrt(act[0]) * ds_bn[0][oc] / sqrt(ds_bn[3][oc] + 0.00001);
        bias[oc] = bias[oc] + sqrt(act[0]) * ds_bn[1][oc];
        // bias[oc] *= sqrt(act[0]);
    }
    act[1] = act[1] / sqrt(act[0]);
}

void update_model_mp(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel, std::vector<double> &bias, std::vector<double> &act) {
    for(int oc = 0; oc < kernel.size(); oc++) {
        for(int ic = 0; ic < kernel[0].size(); ic++) {
            for(int i = 0; i < kernel[0][0].size(); i++) {
                for(int j = 0; j < kernel[0][0][0].size(); j++) {
                    kernel[oc][ic][i][j] = kernel[oc][ic][i][j] * sqrt(act[0]) / 2;
                }
            }
        }
        bias[oc] = bias[oc] * sqrt(act[0]) / 2;
    }
    act[1] = act[1] / (sqrt(act[0]) * 2);
    act[2] = act[2] / 4;
}

void avgpool_fc_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                      std::vector<HEaaN::Message> &output,
                      std::vector<std::vector<double>> &weight, std::vector<double> &bias,
                      HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    const auto log_slots = getLogFullSlots(context);

    for(int i = 0; i < output.size(); i++) {
        fillZero(output[i]);
    }

    // AVG
    HEaaN::Message avg_scale(log_slots);
    fillRandomComplexKernel(avg_scale, 0.015625); // 1/64
    std::vector<HEaaN::Message> avg_msg(input.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < avg_msg.size(); i++) {
        fillZero(avg_msg[i]);
    }

    for(int oc = 0; oc < input.size(); oc++) {
        for(int i = 0; i < input[0].size(); i++) {
            for(int j = 0; j < input[0][0].size(); j++) {
                eval.add(avg_msg[oc], input[oc][i][j], avg_msg[oc]);
            }
        }
        eval.mult(avg_msg[oc], avg_scale, avg_msg[oc]);
    }

    /*
    for(int i = 0; i < avg_msg.size(); i++) {
        printMessage(avg_msg[i]);
    }
    */

    // FC
    std::vector<std::vector<HEaaN::Message>> weight_msg(weight.size(), std::vector<HEaaN::Message>(weight[0].size(), HEaaN::Message(log_slots)));
    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < weight.size(); i++) {
        for(int j = 0; j < weight[0].size(); j++) {
            fillRandomComplexKernel(weight_msg[i][j], weight[i][j]);
        }
    }
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

    std::cout << weight.size() << " " << weight[0].size() << std::endl;

    for(int i = 0; i < weight.size(); i++) {
        for(int j = 0; j < weight[0].size(); j++) {
            HEaaN::Message tmp_msg(log_slots);
            eval.mult(avg_msg[j], weight_msg[i][j], tmp_msg);
            eval.add(output[i], tmp_msg, output[i]);
        }
        eval.add(output[i], bias_msg[i], output[i]);
    }

    /*
    std::cout << "======\n";
    std::cout << "OUTPUT\n";
    std::cout << "======\n";
    for(int i = 0; i < output.size(); i++) {
        printMessage(output[i]);
    }
    */
}

void avgpool_fc_plain_2(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                        std::vector<HEaaN::Message> &output,
                        std::vector<std::vector<double>> &weight, std::vector<double> &bias,
                        HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    const auto log_slots = getLogFullSlots(context);

    for(int i = 0; i < output.size(); i++) {
        fillZero(output[i]);
    }

    // AVG
    HEaaN::Message avg_scale(log_slots);
    fillRandomComplexKernel(avg_scale, 0.0625);    // 1/16
    std::vector<HEaaN::Message> avg_msg(input.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < avg_msg.size(); i++) {
        fillZero(avg_msg[i]);
    }

    for(int oc = 0; oc < input.size(); oc++) {
        for(int i = 0; i < input[0].size(); i++) {
            for(int j = 0; j < input[0][0].size(); j++) {
                eval.add(avg_msg[oc], input[oc][i][j], avg_msg[oc]);
            }
        }
        eval.mult(avg_msg[oc], avg_scale, avg_msg[oc]);
    }

    /*
    for(int i = 0; i < avg_msg.size(); i++) {
        printMessage(avg_msg[i]);
    }
    */

    // FC
    std::vector<std::vector<HEaaN::Message>> weight_msg(weight.size(), std::vector<HEaaN::Message>(weight[0].size(), HEaaN::Message(log_slots)));
    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < weight.size(); i++) {
        for(int j = 0; j < weight[0].size(); j++) {
            fillRandomComplexKernel(weight_msg[i][j], weight[i][j]);
        }
    }
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

    std::cout << weight.size() << " " << weight[0].size() << std::endl;

    for(int i = 0; i < weight.size(); i++) {
        for(int j = 0; j < weight[0].size(); j++) {
            HEaaN::Message tmp_msg(log_slots);
            eval.mult(avg_msg[j], weight_msg[i][j], tmp_msg);
            eval.add(output[i], tmp_msg, output[i]);
        }
        eval.add(output[i], bias_msg[i], output[i]);
    }

    /*
    std::cout << "======\n";
    std::cout << "OUTPUT\n";
    std::cout << "======\n";
    for(int i = 0; i < output.size(); i++) {
        printMessage(output[i]);
    }
    */
}

void fc_plain(std::vector<HEaaN::Message> &input,
              std::vector<HEaaN::Message> &output,
              std::vector<std::vector<double>> &weight, std::vector<double> &bias,
              HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    const auto log_slots = getLogFullSlots(context);

    for(int i = 0; i < output.size(); i++) {
        fillZero(output[i]);
    }

    // FC
    std::vector<std::vector<HEaaN::Message>> weight_msg(weight.size(), std::vector<HEaaN::Message>(weight[0].size(), HEaaN::Message(log_slots)));
    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));

    std::cout << weight.size() << " " << weight[0].size() << std::endl;

    for(int i = 0; i < weight.size(); i++) {
        for(int j = 0; j < weight[0].size(); j++) {
            fillRandomComplexKernel(weight_msg[i][j], weight[i][j]);
        }
    }
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

#pragma omp parallel for collapse(2)
    for(int i = 0; i < weight_msg.size(); i++) {
        for(int j = 0; j < weight_msg[0].size(); j++) {
            HEaaN::Message tmp_msg(log_slots);
            eval.mult(input[j], weight_msg[i][j], tmp_msg);
            eval.add(output[i], tmp_msg, output[i]);
        }
    }
#pragma omp parallel for
    for(int i = 0; i < weight_msg.size(); i++) {
        eval.add(output[i], bias_msg[i], output[i]);
    }
}

void avgpool_fc_cipher(std::vector<HEaaN::Ciphertext> &input_cipher,
                       std::vector<HEaaN::Ciphertext> &output_cipher,
                       std::vector<std::vector<double>> &weight, std::vector<double> &bias,
                       HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
                       HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder, HEaaN::Decryptor &dec, HEaaN::SecretKey &sk) {
    const auto log_slots = getLogFullSlots(context);
    HEaaN::Message zero(log_slots);
    fillZero(zero);
    HEaaN::Message avg_scale(log_slots);
    fillRandomComplexKernel(avg_scale, 0.015625);
    HEaaN::Ciphertext avg_cipher(context);
    enc.encrypt(zero, pack, avg_cipher);
    eval.levelDown(avg_cipher, 5, avg_cipher);

#pragma omp parallel for
    for(int i = 0; i < 64; i++) {
        eval.levelDown(input_cipher[i], 5, input_cipher[i]);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("AVGPOOL ");
// #pragma omp parallel for
    for(int i = 0; i < 64; i++) {
        eval.add(avg_cipher, input_cipher[i], avg_cipher);
    }
    eval.mult(avg_cipher, avg_scale, avg_cipher);
    timer.end();
    timer.print();

    HEaaN::Message dmsg;
    dec.decrypt(avg_cipher, sk, dmsg);
    for(int ii = 0; ii < 64; ii++) {
        std::cout << dmsg[ii].real() << " " ;
    }
    std::cout << std::endl;


    // Encode Kernel
    std::vector<HEaaN::Message> weight_slots(10, HEaaN::Message(log_slots));
#pragma omp parallel for 
    for(int i = 0; i < 10; i++) {
        fillZero(weight_slots[i]);
    }

    int batch_pos = 0;
    while (batch_pos < 32768) {
        for(int i = 0; i < weight.size(); i++) {
            for(int j = 0; j < weight[0].size(); j++) {
                weight_slots[i][batch_pos + j].real(weight[i][j]);
            }
        }
        batch_pos += 64;
    }

    std::vector<HEaaN::Plaintext> weight_plain(10, HEaaN::Plaintext(context));

#pragma omp parallel for
    for(int i = 0; i < 10; i++) {
        weight_plain[i] = edcoder.encode(weight_slots[i]);
        weight_plain[i].setLevel(4);
    }

    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

    timer.start("FC ");
#pragma omp parallel for
    for(int i = 0; i < 10; i++) {
        eval.mult(avg_cipher, weight_plain[i], output_cipher[i]);
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = output_cipher[i];
        for(int p = 5; p >= 0; p--) {
            int rot_pos = pow(2, p);
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(output_cipher[i], tmp_cipher, output_cipher[i]);
            tmp_cipher = output_cipher[i];
        }
        eval.add(output_cipher[i], bias_msg[i], output_cipher[i]);
    }
    timer.end();
    timer.print();
}


void avgpool_fc_cipher_2(std::vector<HEaaN::Ciphertext> &input_cipher,
                         std::vector<HEaaN::Ciphertext> &output_cipher,
                         std::vector<std::vector<double>> &weight, std::vector<double> &bias,
                         HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
                         HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder, HEaaN::Decryptor &dec, HEaaN::SecretKey &sk) {
    const auto log_slots = getLogFullSlots(context);
    HEaaN::Message zero(log_slots);
    fillZero(zero);
    HEaaN::Message avg_scale(log_slots);
    fillRandomComplexKernel(avg_scale, 0.0625); // 1/16
    HEaaN::Ciphertext avg_cipher(context);
    enc.encrypt(zero, pack, avg_cipher);
    eval.levelDown(avg_cipher, 5, avg_cipher);

#pragma omp parallel for
    for(int i = 0; i < 16; i++) {
        eval.levelDown(input_cipher[i], 5, input_cipher[i]);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("AVGPOOL ");
// #pragma omp parallel for
    for(int i = 0; i < 16; i++) {
        eval.add(avg_cipher, input_cipher[i], avg_cipher);
    }
    eval.mult(avg_cipher, avg_scale, avg_cipher);
    timer.end();
    timer.print();

    HEaaN::Message dmsg;
    dec.decrypt(avg_cipher, sk, dmsg);
    for(int ii = 0; ii < 16; ii++) {
        std::cout << dmsg[ii].real() << " " ;
    }
    std::cout << std::endl;


    // Encode Kernel
    std::vector<HEaaN::Message> weight_slots(10, HEaaN::Message(log_slots));
#pragma omp parallel for 
    for(int i = 0; i < 10; i++) {
        fillZero(weight_slots[i]);
    }

    int batch_pos = 0;
    while (batch_pos < 32768) {
        for(int i = 0; i < weight.size(); i++) {
            for(int j = 0; j < weight[0].size(); j++) {
                weight_slots[i][batch_pos + j].real(weight[i][j]);
            }
        }
        batch_pos += 512;
    }

    std::vector<HEaaN::Plaintext> weight_plain(10, HEaaN::Plaintext(context));

#pragma omp parallel for
    for(int i = 0; i < 10; i++) {
        weight_plain[i] = edcoder.encode(weight_slots[i]);
        weight_plain[i].setLevel(4);
    }

    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

    timer.start("FC ");
#pragma omp parallel for
    for(int i = 0; i < 10; i++) {
        eval.mult(avg_cipher, weight_plain[i], output_cipher[i]);
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = output_cipher[i];
        for(int p = 8; p >= 0; p--) {
            int rot_pos = pow(2, p);
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(output_cipher[i], tmp_cipher, output_cipher[i]);
            tmp_cipher = output_cipher[i];
        }
        eval.add(output_cipher[i], bias_msg[i], output_cipher[i]);
    }
    timer.end();
    timer.print();
}

void fc_cipher(std::vector<HEaaN::Ciphertext> &input_cipher,
               std::vector<HEaaN::Ciphertext> &output_cipher,
               std::vector<std::vector<double>> &weight, std::vector<double> &bias,
               int num_class,
               HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor enc,
               HEaaN::HomEvaluator &eval, HEaaN::EnDecoder &edcoder, HEaaN::Decryptor &dec, HEaaN::SecretKey &sk) {
    const auto log_slots = getLogFullSlots(context);
    int level = input_cipher[0].getLevel();

    // Drop Input
    eval.levelDown(input_cipher[0], 4, input_cipher[0]);
    // Encrypt Output
    HEaaN::Message zero(log_slots);
    fillZero(zero);
#pragma omp parallel for
    for(int i = 0; i < weight.size(); i++) {
        enc.encrypt(zero, pack, output_cipher[i]);
        eval.levelDown(output_cipher[i], 3, output_cipher[i]);
    }

    // Encode Kernel
    std::vector<HEaaN::Message> weight_slots(num_class, HEaaN::Message(log_slots));
#pragma omp parallel for 
    for(int i = 0; i < num_class; i++) {
        fillZero(weight_slots[i]);
    }

    int batch_pos = 0;
    while (batch_pos < 32768) {
        for(int i = 0; i < weight.size(); i++) {
            for(int j = 0; j < weight[0].size(); j++) {
                weight_slots[i][batch_pos + j].real(weight[i][j]);
            }
        }
        batch_pos += 512;
    }

    std::vector<HEaaN::Plaintext> weight_plain(num_class, HEaaN::Plaintext(context));

#pragma omp parallel for
    for(int i = 0; i < num_class; i++) {
        weight_plain[i] = edcoder.encode(weight_slots[i]);
        weight_plain[i].setLevel(4);
    }

    std::vector<HEaaN::Message> bias_msg(bias.size(), HEaaN::Message(log_slots));
    for(int i = 0; i < bias.size(); i++) {
        fillRandomComplexKernel(bias_msg[i], bias[i]);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("FC ");
#pragma omp parallel for
    for(int i = 0; i < num_class; i++) {
        eval.mult(input_cipher[0], weight_plain[i], output_cipher[i]);
        HEaaN::Ciphertext tmp_cipher(context);
        tmp_cipher = output_cipher[i];
        for(int p = 8; p >= 0; p--) {
            int rot_pos = pow(2, p);
            // std::cout << "ROT LEFT == " << rot_pos << std::endl;
            eval.leftRotate(tmp_cipher, rot_pos, tmp_cipher);
            eval.add(output_cipher[i], tmp_cipher, output_cipher[i]);
            tmp_cipher = output_cipher[i];
        }
        eval.add(output_cipher[i], bias_msg[i], output_cipher[i]);
    }
    timer.end();
    timer.print();
}

void meanpool_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                    std::vector<std::vector<std::vector<HEaaN::Message>>> &output,
                    HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::cout << "MEAN-POOL\n";
    const auto log_slots = getLogFullSlots(context);
    int kernel_oc = output.size();
    int output_size = output[0].size();

    HEaaN::Message scale_msg(log_slots);
    fillRandomComplexKernel(scale_msg, 0.25);

#pragma omp parallel for collapse(3)
    for(int oc = 0; oc < kernel_oc; oc++) {
        for(int i = 0; i < output_size; i++) {
            for(int j = 0; j < output_size; j++) {
                eval.add( output[oc][i][j], input[oc][2 * i][2 * j], output[oc][i][j]);
                eval.add( output[oc][i][j], input[oc][2 * i][2 * j + 1], output[oc][i][j]);
                eval.add( output[oc][i][j], input[oc][2 * i + 1][2 * j], output[oc][i][j]);
                eval.add( output[oc][i][j], input[oc][2 * i + 1][2 * j + 1], output[oc][i][j]);
                eval.mult(output[oc][i][j], scale_msg, output[oc][i][j]);
            }
        }
    }
    print_pt(output);
}

void meanpool_cipher(std::vector<HEaaN::Ciphertext> &input_cipher, std::vector<HEaaN::Ciphertext> &output_cipher,
                     int row, int col, HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc, HEaaN::HomEvaluator &eval) {
    const auto log_slots = getLogFullSlots(context);
    std::cout << "MEANPOOL\n";

    int new_row = int(row/2);
    int new_col = int(col/2);

    // ENCRYPT OUTPUT
    HEaaN::Message zero(log_slots);
    fillZero(zero);
#pragma omp parallel for
    for(int i = 0; i < new_row * new_col; i++) {
        enc.encrypt(zero, pack, output_cipher[i]);
        eval.levelDown(output_cipher[i], input_cipher[0].getLevel(), output_cipher[i]);
    }

    HEaaN::HEaaNTimer timer(true);
    timer.start("MEANPOOL ");
#pragma omp parallel for collapse(2)
    for(int i = 0; i < new_row; i++) {
        for(int j = 0; j < new_col; j++) {
            int in_pos = 2 * (i * col + j);
            int out_pos = i * new_col + j;
            eval.add(output_cipher[out_pos], input_cipher[in_pos], output_cipher[out_pos]);
            eval.add(output_cipher[out_pos], input_cipher[in_pos + 1], output_cipher[out_pos]);
            eval.add(output_cipher[out_pos], input_cipher[in_pos + col], output_cipher[out_pos]);
            eval.add(output_cipher[out_pos], input_cipher[in_pos + col + 1], output_cipher[out_pos]);
        }
    }
    timer.end();
    timer.print();
}

void FCA_B_1_2(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    // 0
    D[0] = d[0];
    eval.sub(D[0], d[1], D[0]);
    // 1
    D[1] = d[1];
    // 2
    D[2] = d[2];
    eval.sub(D[2], d[1], D[2]);
}

void FCA_B_2_2(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::vector<HEaaN::Ciphertext> tmp_cipher;

    tmp_cipher.push_back(d[1]);
    eval.sub(tmp_cipher[0], d[4], tmp_cipher[0]); // d1 - d4
    tmp_cipher.push_back(d[7]);
    eval.sub(tmp_cipher[1], d[4], tmp_cipher[1]); // d7 - d4

    // 0
    D[0] = d[0];
    eval.sub(D[0], tmp_cipher[0], D[0]);
    eval.sub(D[0], d[3], D[0]);
    // 1
    D[1] = tmp_cipher[0];
    // 2
    D[2] = d[2];
    eval.sub(D[2], tmp_cipher[0], D[2]);
    eval.sub(D[2], d[5], D[2]);
    // 3
    D[3] = d[3];
    eval.sub(D[3], d[4], D[3]);
    // 4
    D[4] = d[4];
    // 5
    D[5] = d[5];
    eval.sub(D[5], d[4], D[5]);
    // 6
    D[6] = d[6];
    eval.sub(D[6], d[3], D[6]);
    eval.sub(D[6], tmp_cipher[1], D[6]);
    // 7
    D[7] = tmp_cipher[1];
    // 8
    D[8] = d[8];
    eval.sub(D[8], tmp_cipher[1], D[8]);
    eval.sub(D[8], d[5], D[8]);

    tmp_cipher.clear();
    tmp_cipher.shrink_to_fit();
}

void FCA_B_3_3(std::vector<HEaaN::Ciphertext> &d, std::vector<HEaaN::Ciphertext> &D, HEaaN::Context &context, HEaaN::HomEvaluator &eval) {
    std::vector<HEaaN::Ciphertext> tmp_cipher;

    tmp_cipher.push_back(d[2]);
    eval.sub(tmp_cipher[0], d[10], tmp_cipher[0]); // d2 - d10
    tmp_cipher.push_back(d[1]);
    eval.sub(tmp_cipher[1], d[9], tmp_cipher[1]);  // d1 - d9
    tmp_cipher.push_back(d[6]);
    eval.add(tmp_cipher[2], d[10], tmp_cipher[2]); // d6 + d10
    tmp_cipher.push_back(d[5]);
    eval.add(tmp_cipher[3], d[9], tmp_cipher[3]);  // d5 + d9
    tmp_cipher.push_back(d[11]);
    eval.sub(tmp_cipher[4], d[9], tmp_cipher[4]);  // d11 - d9
    tmp_cipher.push_back(d[7]);
    eval.sub(tmp_cipher[5], d[5], tmp_cipher[5]);  // d7 - d5
    tmp_cipher.push_back(d[10]);
    eval.sub(tmp_cipher[6], d[6], tmp_cipher[6]);  // d10 - d6
    tmp_cipher.push_back(d[9]);
    eval.sub(tmp_cipher[7], d[5], tmp_cipher[7]);  // d9 - d5
    tmp_cipher.push_back(d[14]);
    eval.sub(tmp_cipher[8], d[6], tmp_cipher[8]);  // d14 - d6
    tmp_cipher.push_back(d[5]);
    eval.sub(tmp_cipher[9], d[13], tmp_cipher[9]); // d5 - d13

    // 0
    D[0] = d[0];
    eval.sub(D[0], tmp_cipher[0], D[0]);
    eval.sub(D[0], d[8], D[0]);
    // 1
    D[1] = tmp_cipher[1];
    eval.add(D[1], tmp_cipher[0], D[1]);
    // 2
    D[2] = tmp_cipher[0];
    eval.sub(D[2], tmp_cipher[1], D[2]);
    // 3
    D[3] = d[3];
    eval.sub(D[3], tmp_cipher[4], D[3]);
    eval.sub(D[3], d[1], D[3]);
    // 4
    D[4] = d[4];
    eval.sub(D[4], tmp_cipher[2], D[4]);
    eval.add(D[4], d[8], D[4]);
    // 5
    D[5] = tmp_cipher[3];
    eval.add(D[5], tmp_cipher[2], D[5]);
    // 6
    D[6] = tmp_cipher[2];
    eval.sub(D[6], tmp_cipher[3], D[6]);
    // 7
    D[7] = tmp_cipher[5];
    eval.add(D[7], tmp_cipher[4], D[7]);
    // 8
    D[8] = d[8];
    eval.sub(D[8], tmp_cipher[6], D[8]);
    eval.sub(D[8], d[4], D[8]);
    // 9
    D[9] = tmp_cipher[7];
    eval.add(D[9], tmp_cipher[6], D[9]);
    // 10
    D[10] = tmp_cipher[6];
    eval.sub(D[10], tmp_cipher[7], D[10]);
    // 11
    D[11] = tmp_cipher[4];
    eval.sub(D[11], tmp_cipher[5], D[11]);
    // 12
    D[12] = d[12];
    eval.sub(D[12], tmp_cipher[8], D[12]);
    eval.sub(D[12], d[4], D[12]);
    // 13
    D[13] = tmp_cipher[8];
    eval.sub(D[13], tmp_cipher[9], D[13]);
    // 14
    D[14] = tmp_cipher[8];
    eval.add(D[14], tmp_cipher[9], D[14]);
    // 15
    D[15] = tmp_cipher[9];
    eval.add(D[15], d[15], D[15]);
    eval.sub(D[15], d[7], D[15]);

    tmp_cipher.clear();
    tmp_cipher.shrink_to_fit();
}

void FCA_A_1_2(int &x, int &y, int &output_row, int &output_col,
        std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
        HEaaN::Context &context, HEaaN::HomEvaluator &eval){
    HEaaN::Ciphertext tmp_cipher(context);
    int pos;

    // 0
    tmp_cipher = DD[0];
    eval.add(tmp_cipher, DD[1], tmp_cipher);
    pos = x * output_col + y;
    out[pos] = tmp_cipher;
    // 1
    tmp_cipher = DD[1];
    eval.add(tmp_cipher, DD[2], tmp_cipher);
    out[pos + 1] = tmp_cipher;
}

void FCA_A_2_2(int &x, int &y, int &output_row, int &output_col,
        std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
        HEaaN::Context &context, HEaaN::HomEvaluator &eval){
    std::vector<HEaaN::Ciphertext> tmp_cipher_vec;
    tmp_cipher_vec.push_back(DD[1]);
    eval.add(tmp_cipher_vec[0], DD[4], tmp_cipher_vec[0]); // DD1 + DD4
    tmp_cipher_vec.push_back(DD[4]);
    eval.add(tmp_cipher_vec[1], DD[7], tmp_cipher_vec[1]);  // DD4 + DD7

    HEaaN::Ciphertext tmp_cipher(context);
    int pos;

    // 0
    tmp_cipher = DD[0];
    eval.add(tmp_cipher, tmp_cipher_vec[0], tmp_cipher);
    eval.add(tmp_cipher, DD[3], tmp_cipher);
    pos = x * output_col + y;
    out[pos] = tmp_cipher;
    // 1
    tmp_cipher = DD[2];
    eval.add(tmp_cipher, tmp_cipher_vec[0], tmp_cipher);
    eval.add(tmp_cipher, DD[5], tmp_cipher);
    out[pos + 1] = tmp_cipher;
    // 2
    tmp_cipher = DD[3];
    eval.add(tmp_cipher, tmp_cipher_vec[1], tmp_cipher);
    eval.add(tmp_cipher, DD[6], tmp_cipher);
    pos = (x + 1) * output_col + y;
    out[pos] = tmp_cipher;
    // 3
    tmp_cipher = DD[5];
    eval.add(tmp_cipher, tmp_cipher_vec[1], tmp_cipher);
    eval.add(tmp_cipher, DD[8], tmp_cipher);
    out[pos + 1] = tmp_cipher;

    tmp_cipher_vec.clear();
    tmp_cipher_vec.shrink_to_fit();
}

void FCA_A_3_3(int &x, int &y, int &output_row, int &output_col,
        std::vector<HEaaN::Ciphertext> &DD, std::vector<HEaaN::Ciphertext> &out, 
        HEaaN::Context &context, HEaaN::HomEvaluator &eval){
    int pos;
    std::vector<HEaaN::Ciphertext> tmp_cipher_vec;
    HEaaN::Ciphertext tmp_cipher(context);

    tmp_cipher_vec.push_back(DD[4]);
    eval.add(tmp_cipher_vec[0], DD[5], tmp_cipher_vec[0]);
    eval.add(tmp_cipher_vec[0], DD[6], tmp_cipher_vec[0]);
    tmp_cipher_vec.push_back(DD[8]);
    eval.add(tmp_cipher_vec[1], DD[9], tmp_cipher_vec[1]);
    eval.add(tmp_cipher_vec[1], DD[10], tmp_cipher_vec[1]);
    tmp_cipher_vec.push_back(DD[5]);
    eval.sub(tmp_cipher_vec[2], DD[6], tmp_cipher_vec[2]);
    eval.add(tmp_cipher_vec[2], DD[7], tmp_cipher_vec[2]);
    tmp_cipher_vec.push_back(DD[9]);
    eval.sub(tmp_cipher_vec[3], DD[10], tmp_cipher_vec[3]);
    eval.add(tmp_cipher_vec[3], DD[11], tmp_cipher_vec[3]);
    
    /*
    HEaaN::Ciphertext tmp_cipher(context), tmp_cipher_1(context), tmp_cipher_2(context), tmp_cipher_3(context), tmp_cipher_4(context);

    tmp_cipher_1 = DD[4];
    eval.add(tmp_cipher_1, DD[5], tmp_cipher_1);
    eval.add(tmp_cipher_1, DD[6], tmp_cipher_1);
    tmp_cipher_2 = DD[8];
    eval.add(tmp_cipher_2, DD[9], tmp_cipher_2);
    eval.add(tmp_cipher_2, DD[10], tmp_cipher_2);
    tmp_cipher_3 = DD[5];
    eval.sub(tmp_cipher_3, DD[6], tmp_cipher_3);
    eval.add(tmp_cipher_3, DD[7], tmp_cipher_3);
    tmp_cipher_4 = DD[9];
    eval.sub(tmp_cipher_4, DD[10], tmp_cipher_4);
    eval.add(tmp_cipher_4, DD[11], tmp_cipher_4);
    */

    // 0
    tmp_cipher = DD[0];
    eval.add(tmp_cipher, DD[1], tmp_cipher);
    eval.add(tmp_cipher, DD[2], tmp_cipher);
    eval.add(tmp_cipher, tmp_cipher_vec[0], tmp_cipher);
    eval.add(tmp_cipher, tmp_cipher_vec[1], tmp_cipher);
    pos = x * output_col + y;
    out[pos] = tmp_cipher;
    // 1
    tmp_cipher = DD[1];
    eval.sub(tmp_cipher, DD[2], tmp_cipher);
    eval.add(tmp_cipher, DD[3], tmp_cipher);
    eval.add(tmp_cipher, tmp_cipher_vec[2], tmp_cipher);
    eval.add(tmp_cipher, tmp_cipher_vec[3], tmp_cipher);
    out[pos + 1] = tmp_cipher;
    // 2
    tmp_cipher = DD[12];
    eval.add(tmp_cipher, tmp_cipher_vec[0], tmp_cipher);
    eval.sub(tmp_cipher, tmp_cipher_vec[1], tmp_cipher);
    eval.add(tmp_cipher, DD[13], tmp_cipher);
    eval.add(tmp_cipher, DD[14], tmp_cipher);
    pos = (x + 1) * output_col + y;
    out[pos] = tmp_cipher;
    // 3
    tmp_cipher = DD[13];
    eval.add(tmp_cipher, tmp_cipher_vec[2], tmp_cipher);
    eval.sub(tmp_cipher, tmp_cipher_vec[3], tmp_cipher);
    eval.sub(tmp_cipher, DD[14], tmp_cipher);
    eval.add(tmp_cipher, DD[15], tmp_cipher);
    out[pos + 1] = tmp_cipher;

    tmp_cipher_vec.clear();
    tmp_cipher_vec.shrink_to_fit();
}
