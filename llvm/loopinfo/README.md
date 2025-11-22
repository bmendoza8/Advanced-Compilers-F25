LoopInfoExamples.cpp was copied and updated to work with my llvm and to complete,
a) print out more info such as loop preheader and latch. b) Recursively call
getSubLoops() to print the loops in a loop nest along with their depths. c) Any
other loop information you would like to get. 

Changes include: 
-empty() was replaced with getSubLoops().empty() since Im using llvm 18
and apparently it does not use empty(). 
-Adding a recurive function to bypass run() which prints a summary of the 
llvm loop. All nested loops are also traversed recursively and prints them with
adequate indentation to provide a strcutre we can follow.

Test:
cd ~/Advanced-Compilers-F25/llvm/loopinfo/build

cat > loops_test.c << 'EOF'
int sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}
EOF

/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm loops_test.c -o loops_test.ll

/usr/lib/llvm-18/bin/opt \
  -passes=mem2reg,loop-simplify,loop-rotate \
  loops_test.ll \
  -S -o loops_test_canonical.ll

Run:
/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./LoopInfoExample.so \
  -passes="loop-info-example" \
  loops_test_canonical.ll \
  -disable-output

