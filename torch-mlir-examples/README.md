I could not get torch-mlir installed from source. I tried the way you said in
the slide on installing torch and it did not work. I tried another way I found on
YouTube but it also did not work. I looked up if llvm 18 could be the problem 
but everything said it should work. Then I pasted these errors onto Claude
which helps with software/code/computer errors and basically told me that my
machine was not generating "Python binding that torch-mlir expects" and that
the easiest way to fix was to use a Mac with Homebrew which I can't afford. It
also stated that the program should run on a machine that its able to use 
torch mlir. These are the errors:
-Dpybind11_DIR to change) -- Found pybind11 v3.0.1: /home/bmendoza8/Advanced-Co
mpilers-F25/torch-mlir-examples/venv/lib/python3.12/site-packages/pybind1
1/include -- Python prefix = '', suffix = '', extension = '.cpython-312-x86_64
-linux-gnu.so CMake Warning at externals/stablehlo/stablehlo/integrations/cpp
/builder/CMakeLists.txt:144 (message): gtest not found, unittests will not be 
available. 

CMake Error at /usr/lib/llvm-18/lib/cmake/mlir/AddMLIRPython.cmake:530 
(get_target_property): get_target_property() called with non-existent target
 "MLIRPythonSources". Call Stack (most recent call first): /usr/lib/llvm-18/lib/
cmake/mlir/AddMLIRPython.cmake:480 (_flatten_mlir_python_targets)
PLUS AT LOT MORE(about 20 errors)
