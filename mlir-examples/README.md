For MLIR-Examples assignment everything worked pretty straight foward, except for
the python part. Since Im using ubuntu on windows(using linux), pip install numpy
did not work, so I tried sudo apt install python3.12-venv, which for some reason
did not work at first. I came back to the project later and it worked fine, so if
you have issues(embedded onto the files maybe) let me know. 

I also had to add /usr/lib/llvm-18/bin manually, which Im assuming is because 
of me running llvm 18(Again, I cannot get LLVM 21+ to work at all)

For MLIR it was installed through LLVM as stated in the article.
I had to change "...add.dylib" to "...add.so" in array_add.py to make it work
on Ubuntu.

Building and running MLIR examples:
mlir-opt array_add.mlir \
  -convert-scf-to-cf \
  -convert-arith-to-llvm \
  -convert-cf-to-llvm \
  -finalize-memref-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o array_add_opt.mlir

Convert MLIR to LLVM IR:
mlir-translate array_add_opt.mlir -mlir-to-llvmir -o array_add_opt.ll

Complie LLVM IR:
llc -filetype=obj --relocation-model=pic array_add_opt.ll -o array_add_opt.o

Create library:
clang -shared -fPIC array_add_opt.o -o libarray_add.so

Running both 'python array_add.py' and 'python array_add_jit.py' the output is
identical which means MLIR to LLVM lowering worked as expected. 
