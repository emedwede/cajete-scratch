#!/bin/bash

rm -r -f cpu-build
mkdir cpu-build

cd cpu-build

#Same path as on blazar to make life easier
cmake -DCMAKE_PREFIX_PATH="~/ExtraLibs/CajeteDepend/Cabana/cpu-build/install" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

make -j4
