For DefUseChains, I used the llvm that worked for the previous assignment, 
hence why I am using the same folder as the previous one. mkdir to make the 
defuse, LoopInfoExample has its own 'folder' under llvm.
As simple test is:

cd ~/Advanced-Compilers-F25/llvm/defuse/build

cat > defuse_test.c << 'EOF'
int foo(int a, int b) {
    int c = a + b;
    int d = c * 2;
    return d;
}
EOF

/usr/lib/llvm-18/bin/clang \
  -O0 -Xclang -disable-O0-optnone \
  -S -emit-llvm defuse_test.c -o defuse_test.ll


THEN, RUN:

/usr/lib/llvm-18/bin/opt \
  -load-pass-plugin=./DefUseChains.so \
  -passes="def-use-chains" \
  defuse_test.ll \
  -disable-output
You should now see the block labels, and were [DEF] <instruction> is used,
along side the function name.
