#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

using namespace std;

extern double tmp_time;
extern double total_save_model_time;
extern double total_load_model_time;
extern double total_offline_time;
extern double total_online_time;
extern double save_model_time;
extern double load_model_time;
extern double offline_time;
extern double online_time;
extern double vm, vm2, rss, rss2;

void process_mem_usage(double& vm_usage, double& resident_set) {
    vm_usage = 0.0;
    resident_set = 0.0;

    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;

    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    std::cout << "CHECK RAM USAGE ";
    std::cout << "VM: " << vm_usage << "; RSS: " << resident_set << std::endl;
}

template<class T>
vector<vector<T>> read_winograd_matrix(string name, int output_pile_size, int kernel_size) {
    int width, height;

    if(name == "AT") {
        width = output_pile_size;
        height = output_pile_size + kernel_size - 1;
    }else if(name == "BT") {
        width = output_pile_size + kernel_size - 1;
        height = output_pile_size + kernel_size - 1;
    }else if (name == "G") {
        width = output_pile_size + kernel_size - 1;
        height = kernel_size;
    }

    vector<vector<T>> M(width, vector<T>(height));

    string inFileName = "../../data/" + name + to_string(output_pile_size) + 
                        "x" + to_string(kernel_size) + ".txt";
    ifstream inFile;
    inFile.open(inFileName.c_str());

    double tmp;
    if(inFile.is_open()) {
        for(int i = 0; i < width; i++) {
            for(int j = 0; j < height; j++) {
                inFile >> tmp;
                M[i][j] = tmp;
            }
        }
        inFile.close(); // CLose input file
    }
    else { //Error message
        cerr << "Can't find input file " << inFileName << endl;
    }

    return M;
}

template <class T>
void transpose_matrix(vector<vector<T>> &A, vector<vector<T>> &M) {
    for(int i = 0; i < A[0].size(); i++) {
        for(int j = 0; j < A.size(); j++) {
            M[i][j] = A[j][i];
        }
    }
}

template<class T>
vector<vector<T>> matmul(vector<vector<T>> &A, vector<vector<T>> &B) {
    int A_row = A.size();
    int A_col = A[0].size();
    int B_row = B.size();
    int B_col = B[0].size();

    vector<vector<T>> M(A_row, vector<T>(B_col, 0));

    for(int i = 0; i < A_row; i++) {
        for(int j = 0; j < B_col; j++) {
            for(int k = 0; k < A_col; k++) {
                M[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return  M;
}

template<class T>
void split_matrix(vector<T> &input,
                  vector<T> &I,
                  string type,
                  int input_row,
                  int input_col,
                  int num_input_row,
                  int num_input_col) {

#pragma omp parallel for collapse(2)
    for(int i = 0; i < input_row; i++) {
        for(int j = 0; j < input_col; j++) {
            int input_pos = i * input_col + j;
            int x = i/2;
            int y = j/2;
            if(i % 2 == 0) {
                if(j % 2 == 0) {
                    if(type == "A") {
                        int output_pos = x * num_input_col + y;
                        I[output_pos] = input[input_pos];
                        // IA[x][y][ic] = input[i][j][ic];
                    }
                }else{
                    if(type == "B") {
                        int output_pos = y * num_input_col + x;
                        I[output_pos] = input[input_pos]; // transpose
                                                          // IB[y][x][ic] = input[i][j][ic]; // transpose
                    }
                }
            }else{
                if(j % 2 == 0) {
                    if(type == "C") {
                        int output_pos = x * num_input_col + y;
                        I[output_pos] = input[input_pos];
                        // IC[x][y][ic] = input[i][j][ic];
                    }
                }else{
                    if(type == "D") {
                        int output_pos = x * num_input_col + y;
                        I[output_pos] = input[input_pos];
                    }
                }
            }
        }
    }
}


template<class T>
void split_kernel(vector<vector<vector<vector<T>>>> &W,
                  vector<vector<vector<vector<T>>>> &KA,
                  vector<vector<vector<vector<T>>>> &KB,
                  vector<vector<vector<vector<T>>>> &KC,
                  vector<vector<vector<vector<T>>>> &KD) {
#pragma omp parallel for collapse(4)
    for(int kc = 0; kc < W.size(); kc++) {
        for(int ic = 0; ic < W[0].size(); ic++) {
            for(int i = 0; i < W[0][0].size(); i++) {
                for(int j = 0; j < W[0][0][0].size(); j++) {
                    int x = i/2;
                    int y = j/2;
                    if(i % 2 == 0) {
                        if(j % 2 == 0) {
                            KA[kc][ic][x][y] = W[kc][ic][i][j];
                        }else{
                            KB[kc][ic][y][x] = W[kc][ic][i][j]; // transpose
                        }
                    }else{
                        if(j % 2 == 0) {
                            KC[kc][ic][x][y] = W[kc][ic][i][j];
                        }else{
                            KD[kc][ic][x][y] = W[kc][ic][i][j];
                        }
                    }
                }
            }
        }
    }
}

