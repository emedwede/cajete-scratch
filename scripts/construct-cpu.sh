#!/bin/bash

rm -r -f cpu-build
mkdir cpu-build

cd cpu-build

cmake -DCMAKE_PREFIX_PATH="/home/erock/ExtraLibs/CajeteDepend/Cabana/cpu-build/install" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

make -j4
