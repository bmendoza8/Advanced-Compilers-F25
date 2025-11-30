Project is based on LLVM 21 but once again, I cannot get LLVM 21 to work for me.
I read and watched videos but I can't seem to get it to work, which makes me
think that I might be because of something else I might have installed in my 
computer.

Since Im using llvm 18, I had to make slight modifications to some files, nothing
extensive. 
For DynamicCallCounter I changed 'getWithCaptureInfo' to 'getContext' and 
'empty' statements. 
No changes to fit llvm 18 to AffineRecurrence as it worked using the provided
file.

How to use:
To build .so files use ->

cd ~/llvm-tutor
mkdir -p build
cd build

cmake -DLT_LLVM_INSTALL_DIR=/usr/lib/llvm-18 ..
make

To generate LLVM IR and canonical form(matmul) ->
cd ~/llvm-tutor

/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -emit-llvm -S inputs/matmul.c \
  -o inputs/matmul.ll

/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg,loop-simplify,loop-rotate \
  inputs/matmul.ll -S -o inputs/matmul_canonical.ll

To run SimpleLICM ->
cd ~/llvm-tutor/build
mkdir -p outputs

/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./lib/libSimpleLICM.so \
  -passes=simple-licm \
  -S -o outputs/matmul_licm.ll \
  ../inputs/matmul_canonical.ll

To see IR from SimpleLICM(use 'q' to exit, I learned this the hard way) ->
less outputs/matmul_licm.ll

To run AffineRecurrence ->
cd ~/llvm-tutor/build

/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./lib/libAffineRecurrence.so \
  -passes=affine-recurrence \
  ../inputs/matmul_canonical.ll \
  -disable-output

To analyze and eliminate(DerivedInductionVar) ->
cd ~/llvm-tutor/build
mkdir -p outputs

/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./lib/libDerivedInductionVar.so \
  -passes=derived-iv \
  ../inputs/matmul_canonical.ll \
  -S -o outputs/matmul_derivediv.ll



