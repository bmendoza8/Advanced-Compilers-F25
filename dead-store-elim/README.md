For Dead Store Elimination, you can build the plug in by using:

cd Advanced-Compilers-F25/dead-store-elim

/usr/lib/llvm-18/bin/clang++ \
  -std=c++17 -fPIC -shared MemorySSADemo.cpp \
  -o libMemorySSADemo.so \
  $(llvm-config --cxxflags --ldflags --system-libs --libs core analysis)

Then to compile i used clang from llvm 18:
demo.c:
/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm demo.c -o demo.ll
/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg \
  demo.ll -S -o demo_simplified.ll

demo2.c:
/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm demo2.c -o demo2.ll
/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg \
  demo2.ll -S -o demo2_simplified.ll

test1.c:
/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm test1.c -o test1.ll
/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg \
  test1.ll -S -o test1_simplified.ll

array_loop 1.c:
/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm "array_loop 1.c" -o array_loop1.ll
/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg \
  array_loop1.ll -S -o array_loop1_simplified.ll

branch_example.c:
/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm branch_example.c -o branch_example.ll
/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg \
  branch_example.ll -S -o branch_example_simplified.ll

To run MemorySSA:
/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./libMemorySSADemo.so \
  -passes="memssa-demo" \
  demo2_simplified.ll \
  -disable-output

and this outputs a DOT file:
/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./libMemorySSADemo.so \
  -passes="memssa-demo" \
  demo2_simplified.ll \
  -disable-output 2> memssa_demo2.dot

To run the DSE pass, use:
/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./libMemorySSADemo.so \
  -passes="simple-dse" \
  demo_simplified.ll \
  -S -o demo_dse.ll

