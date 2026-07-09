################################################################################
##                                                                            ##
## Copyright (C) 2021-2022 Crypto Lab Inc.                                    ##
##                                                                            ##
## - This file is part of HEaaN homomorphic encryption library.               ##
## - HEaaN cannot be copied and/or distributed without the express permission ##
##  of Crypto Lab Inc.                                                        ##
##                                                                            ##
################################################################################

project(HEaaN_examples LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # clangd


find_package(OpenMP REQUIRED)
find_package(HEaaN REQUIRED)
find_package(HighFive REQUIRED)
# find_package(CUDA REQUIRED)

# add_executable(ex1_encrypt-decrypt 1_encrypt-decrypt.cpp)
# target_link_libraries(ex1_encrypt-decrypt PUBLIC HEaaN::HEaaN)

# add_executable(ex2_evalpoly 2_evalpoly.cpp)
# target_link_libraries(ex2_evalpoly PUBLIC HEaaN::HEaaN)

# add_executable(ex3_average 3_average.cpp)
# target_link_libraries(ex3_average PUBLIC HEaaN::HEaaN)

# add_executable(ex4_save-load 4_save-load.cpp)
# target_link_libraries(ex4_save-load PUBLIC HEaaN::HEaaN)

# add_executable(ex5_bootstrap 5_bootstrap.cpp)
# target_link_libraries(ex5_bootstrap PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX)

# add_executable(ex6_bootstrap_gpu 6_bootstrap_gpu.cpp)
# target_link_libraries(ex6_bootstrap_gpu PUBLIC HEaaN::HEaaN)

add_executable(ex8_resnet 8_resnet.cpp)
target_link_libraries(ex8_resnet PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive)
# target_link_libraries(ex8_resnet PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive ${CUDA_LIBRARIES})

add_executable(ex9_vgg11 9_vgg11.cpp)
target_link_libraries(ex9_vgg11 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive)
# target_link_libraries(ex9_vgg11 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive ${CUDA_LIBRARIES})

add_executable(ex10_vgg16 10_vgg16.cpp)
target_link_libraries(ex10_vgg16 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive)
# target_link_libraries(ex10_vgg16 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive ${CUDA_LIBRARIES})

add_executable(ex11_conv 11_conv.cpp)
target_link_libraries(ex11_conv PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive)
# target_link_libraries(ex11_conv PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive ${CUDA_LIBRARIES})

add_executable(ex13_resnet18 13_resnet18.cpp)
target_link_libraries(ex13_resnet18 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive)
# target_link_libraries(ex13_resnet18 PUBLIC HEaaN::HEaaN OpenMP::OpenMP_CXX HighFive::HighFive ${CUDA_LIBRARIES})
