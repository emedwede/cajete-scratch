#!/bin/bash

rm -r -f gpu-build

mkdir gpu-build

cd gpu-build
HOME_PATH=~/
COMPILER_PATH=$HOME_PATH"ExtraLibs/CajeteDepend/kokkos/bin/nvcc_wrapper"
echo $COMPILER_PATH
cmake -DCMAKE_PREFIX_PATH="~/ExtraLibs/CajeteDepend/Cabana/gpu-build/install" \
    -DCMAKE_CXX_COMPILER=$COMPILER_PATH \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

make -j4
