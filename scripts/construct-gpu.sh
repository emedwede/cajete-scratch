#!/bin/bash

rm -r -f gpu-build

mkdir gpu-build

cd gpu-build

cmake -DCMAKE_PREFIX_PATH="/home/erock/ExtraLibs/CajeteDepend/Cabana/gpu-build/install" \
    -DCMAKE_CXX_COMPILER="/home/erock/ExtraLibs/CajeteDepend/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

make -j4
