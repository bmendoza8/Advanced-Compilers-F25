This is not a typical README, as I wanted to document what I had to change
in order for LLVM to work for me. I tried using LLVM 21 as I thought it would
be better but I had a lot of problems setting it so I stuck with LLVM but
since I had already started with 21, it led to even more issues. 

I used to commands and files from github to install 18.1.3 as well as some
implementations of my own. 
The git clone https://github.com/banach-space/llvm-tutor.git did not work as 
it expected LLVM 21, so I changed some lines in order for it to work. 
After I got it to run, I used 
In ~/llvm-tutor:
rm -rf build
mkdir build
cd build
export CC=/usr/lib/llvm-18/bin/clang
export CXX=/usr/lib/llvm-18/bin/clang++

cmake -DLT_LLVM_INSTALL_DIR=/usr/lib/llvm-18 -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX ..
make
to build the LLVM-tutor.

build MyPass:
cmake -DLLVM_DIR=$(llvm-config --cmakedir) ..
make

Run MyPass:
opt -load-pass-plugin=./MyPass.so -passes="my-pass" test.ll -disable-output

Build llvm-tutor:
cmake -DLT_LLVM_INSTALL_DIR=/usr/lib/llvm-18 ..
make

Run llvm-tutor passes:
opt -load-pass-plugin=./lib/libHelloWorld.so -passes=hello-world input.ll 
	-disable-output 
