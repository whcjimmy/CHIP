#include <iostream>
#include <optional>
#include <random>
#include <unistd.h>
#include "omp.h"

// #include "HEaaN/heaan.hpp"
#include "HEaaN/HEaaN.hpp"

std::default_random_engine gen{std::random_device()()};

inline long double randNum() {
    std::uniform_real_distribution<long double> dist(-1.0L, 1.0L);
    return dist(gen);
}

void fillRandomComplex(HEaaN::Message &msg) {
#pragma omp parallel for 
    for (size_t i = 0; i < msg.getSize(); ++i) {
        msg[i].real(randNum());
        // msg[i].real(0.001);
        msg[i].imag(0);
    }
}

void fillRandomComplexKernel(HEaaN::Message &msg, double rand) {
    for (size_t i = 0; i < msg.getSize(); ++i) {
        msg[i].real(rand);
        msg[i].imag(0);
    }
}

void fillRandomReal(HEaaN::Message &msg, std::optional<size_t> num) {
    size_t length = num.has_value() ? num.value() : msg.getSize();
    size_t idx = 0;
    for (; idx < length; ++idx) {
        msg[idx].real(randNum());
        msg[idx].imag(0.0);
    }
    // If num is less than the size of msg,
    // all remaining slots are zero.
    for (; idx < msg.getSize(); ++idx) {
        msg[idx].real(0.0);
        msg[idx].imag(0.0);
    }
}

void fillZero(HEaaN::Message &msg) {
#pragma omp parallel for 
    for (size_t i = 0; i < msg.getSize(); ++i) {
        msg[i].real(0.0);
        msg[i].imag(0.0);
    }
}

void msgAdd(HEaaN::Message &msg1, HEaaN::Message &msg2) {
    for (size_t i = 0; i < msg1.getSize(); ++i) {
        msg1[i].real(msg1[i].real() + msg2[i].real());
    }
}

void printMessage(const HEaaN::Message &msg, bool is_complex = true,
                  size_t start_num = 2, size_t end_num = 2) {
    const size_t msg_size = msg.getSize();

    std::cout << "[ ";
    for (size_t i = 0; i < start_num; ++i) {
        if (is_complex)
            std::cout << msg[i] << ", ";
        else
            std::cout << msg[i].real() << ", ";
    }
    std::cout << "..., ";
    for (size_t i = end_num; i > 1; --i) {
        if (is_complex)
            std::cout << msg[msg_size - i] << ", ";
        else
            std::cout << msg[msg_size - i].real() << ", ";
    }
    if (is_complex)
        std::cout << msg[msg_size - 1] << " ]" << std::endl;
    else
        std::cout << msg[msg_size - 1].real() << " ]" << std::endl;
}

std::string presetNamer(const HEaaN::ParameterPreset preset) {
    switch (preset) {
    case HEaaN::ParameterPreset::FVa:
        return "FVa";
    case HEaaN::ParameterPreset::FVb:
        return "FVb";
    case HEaaN::ParameterPreset::FGa:
        return "FGa";
    case HEaaN::ParameterPreset::FGb:
        return "FGb";
    case HEaaN::ParameterPreset::FTa:
        return "FTa";
    case HEaaN::ParameterPreset::FTb:
        return "FTb";
    case HEaaN::ParameterPreset::ST19:
        return "ST19";
    case HEaaN::ParameterPreset::ST14:
        return "ST14";
    case HEaaN::ParameterPreset::ST11:
        return "ST11";
    case HEaaN::ParameterPreset::ST8:
        return "ST8";
    case HEaaN::ParameterPreset::ST7:
        return "ST7";
    case HEaaN::ParameterPreset::SS7:
        return "SS7";
    case HEaaN::ParameterPreset::SD3:
        return "SD3";
    case HEaaN::ParameterPreset::CUSTOM:
        return "CUSTOM";
    case HEaaN::ParameterPreset::FVc:
        return "FVc";
    default:
        throw std::invalid_argument("Not supported parameter");
    }
}

void generate_1D_data_double(std::vector<double> &data) {
#pragma omp parallel for 
    for(int i = 0; i < data.size(); i++) {
        data[i] = 0;
    }
}

void generate_3D_data(std::vector<std::vector<std::vector<HEaaN::Message>>> &data) {
#pragma omp parallel for collapse(3)
    for(int ic = 0; ic < data.size(); ic++) {
        for(int i = 0; i < data[0].size(); i++) {
            for(int j = 0; j < data[0][0].size(); j++) {
                fillRandomComplexKernel(data[ic][i][j], 1);
                // fillRandomComplex(data[ic][i][j]);
            }
        }
    }
}

void generate_1D_zero(std::vector<HEaaN::Message> &data) {
#pragma omp parallel for
    for(int i = 0; i < data.size(); i++) {
        fillZero(data[i]);
    }
}

void generate_3D_zero(std::vector<std::vector<std::vector<HEaaN::Message>>> &data) {
#pragma omp parallel for collapse(3)
    for(int ic = 0; ic < data.size(); ic++) {
        for(int i = 0; i < data[0].size(); i++) {
            for(int j = 0; j < data[0][0].size(); j++) {
                fillZero(data[ic][i][j]);
            }
        }
    }
}

void generate_kernel_bias(std::vector<std::vector<std::vector<std::vector<double>>>> &kernel, std::vector<double> &bias) {
#pragma omp parallel for collapse(4)
    for(int oc = 0; oc < kernel.size(); oc++) {
        for(int ic = 0; ic < kernel[0].size(); ic++) {
            for(int i = 0; i < kernel[0][0].size(); i++) {
                for(int j = 0; j < kernel[0][0][0].size(); j++) {
                    kernel[oc][ic][i][j] = 0.01;
                    // kernel[oc][ic][i][j] = randNum() * 0.1;
                }
            }
        }
    }
#pragma omp parallel for 
    for(int oc = 0; oc < bias.size(); oc++) {
        bias[oc] = 0.01;
        // bias[oc] = randNum();
    }
}

void resize_1d(std::vector<HEaaN::Message> &input, int size, HEaaN::Context &context) {
    const auto log_slots = getLogFullSlots(context);
    input.resize(size, HEaaN::Message(log_slots));
    generate_1D_zero(input);
}

void resize_3d(std::vector<std::vector<std::vector<HEaaN::Message>>> &input, int row_size, int col_size, int chnl_size, 
               HEaaN::Context &context) {
    const auto log_slots = getLogFullSlots(context);
    input.resize(chnl_size);
    for (auto& row : input) {
        row.resize(row_size);
        for (auto& col : row) {
            col.resize(col_size, HEaaN::Message(log_slots));
        }
    }
    generate_3D_zero(input);
}

void resize_4d(std::vector<std::vector<std::vector<std::vector<HEaaN::Message>>>> &input, int row_size, int col_size, int ic_size, int oc_size,
               HEaaN::Context &context) {
    const auto log_slots = getLogFullSlots(context);
    input.resize(oc_size);
    for(auto& ic: input) {
        ic.resize(ic_size);
        for (auto& row : ic) {
            row.resize(row_size);
            for (auto& col : row) {
                col.resize(col_size, HEaaN::Message(log_slots));
            }
        }
    }
}

void copy_pt_1d(std::vector<HEaaN::Message> &input,
                std::vector<HEaaN::Message> &output) {
#pragma omp parallel for
    for(int i = 0; i < output.size(); i++) {
        input[i] = output[i];
    }
}

void copy_pt(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
             std::vector<std::vector<std::vector<HEaaN::Message>>> &output) {
#pragma omp parallel for collapse(3)
    for(int oc = 0; oc < output.size(); oc++) {
        for(int i = 0; i < output[0].size(); i++) {
            for(int j = 0; j < output[0][0].size(); j++) {
                input[oc][i][j] = output[oc][i][j];
            }
        }
    }
}

void copy_ct(std::vector<HEaaN::Ciphertext> &input_cipher, std::vector<HEaaN::Ciphertext> &output_cipher, int row, int col) {
#pragma omp parallel for
    for(int i = 0; i < row * col; i++) {
        input_cipher[i] = output_cipher[i];
    }
}

void print_pt(std::vector<std::vector<std::vector<HEaaN::Message>>> &output) {
    for(int oc = 0; oc < output.size(); oc++) {
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < output[0].size(); i++) {
                for(int j = 0; j < output[0][0].size(); j++) {
                    std::cout << output[oc][i][j][0].real() << "\t";
                }
                std::cout << std::endl;
            }
        }
    }
}

void print_ct(std::vector<HEaaN::Ciphertext> &output_cipher, 
              int row, int col, int rp_pack_size, int kernel_oc, std::string type, 
              HEaaN::Decryptor &dec, HEaaN::SecretKey &sk) {
    HEaaN::Message dmsg;
    for(int oc = 0; oc < kernel_oc; oc++){
        if(oc == 0) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < row; i++) {
                for(int j = 0; j < col; j++) {
                    int output_pos = i * col + j;
                    int chnl_pos = (type == "A") ? oc : oc * rp_pack_size; 
                    dec.decrypt(output_cipher[output_pos], sk, dmsg);
                    std::cout << dmsg[chnl_pos].real() << "\t" ;
                }
                std::cout << std::endl;
            }
        }
    }
}

void print_ct_2(std::vector<HEaaN::Ciphertext> &output_cipher, 
              int row, int col, int rp_pack_size, int kernel_oc, std::string type, 
              HEaaN::Decryptor &dec, HEaaN::SecretKey &sk) {
    HEaaN::Message dmsg;
    for(int oc = 0; oc < kernel_oc; oc++){
        // if(oc < 6) {
            std::cout << "oc " << oc << std::endl;
            for(int i = 0; i < row; i++) {
                for(int j = 0; j < col; j++) {
                    int output_pos = i * col + j;
                    int chnl_pos = (type == "A") ? oc : oc * rp_pack_size; 
                    dec.decrypt(output_cipher[output_pos], sk, dmsg);
                    std::cout << dmsg[chnl_pos].real() << "\t" ;
                }
                std::cout << std::endl;
            }
        // }
    }
}

void padding_plain(std::vector<std::vector<std::vector<HEaaN::Message>>> &input,
                   std::vector<std::vector<std::vector<HEaaN::Message>>> &input_pad,
                   int size, int kernel_ic) {
    generate_3D_zero(input_pad);
#pragma omp parallel for collapse(3)
    for(int ic = 0; ic < kernel_ic; ic++) {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                input_pad[ic][i + 1][j + 1] = input[ic][i][j];
            }
        }
    }
}

void padding_cipher(std::vector<HEaaN::Ciphertext> &vec_cipher_1,
                    std::vector<HEaaN::Ciphertext> &vec_cipher_2,
                    int size, int stride,
                    HEaaN::Context &context, HEaaN::KeyPack &pack, HEaaN::Encryptor &enc, HEaaN::HomEvaluator &eval) {
    const auto log_slots = getLogFullSlots(context);
    int level = vec_cipher_2[0].getLevel();
    HEaaN::Message zero(log_slots);
    fillZero(zero);

    int pad_size = (stride == 1) ? size + 2 : size + 1;
#pragma omp parallel for collapse(2)
    for(int i = 0; i < pad_size; i++) {
        for(int j = 0; j < pad_size; j++) {
            int pad_pos = i * pad_size + j;
            if(stride == 1) {
                if(i == 0 || i == (pad_size - 1) || j == 0 || j == (pad_size - 1)) {
                    enc.encrypt(zero, pack, vec_cipher_1[pad_pos]);
                    eval.levelDown(vec_cipher_1[pad_pos], level, vec_cipher_1[pad_pos]);
                }else{
                    int old_pos = (i - 1) * size + (j - 1);
                    vec_cipher_1[pad_pos] = vec_cipher_2[old_pos];
                }
            }else{
                if(i == 0 || j == 0) {
                    enc.encrypt(zero, pack, vec_cipher_1[pad_pos]);
                    eval.levelDown(vec_cipher_1[pad_pos], level, vec_cipher_1[pad_pos]);
                }else{
                    int old_pos = (i - 1) * size + (j - 1);
                    vec_cipher_1[pad_pos] = vec_cipher_2[old_pos];
                }
            }
        }
    }
}

