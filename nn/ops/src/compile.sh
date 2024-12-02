# nvcc -c matmul.cu -o matmul.o -x cu -Xcompiler -fPIC
# g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include matmul_wrapper.cpp matmul.o -o matmul$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart

# nvcc -c matadd.cu -o matadd.o -x cu -Xcompiler -fPIC
# g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include matadd_wrapper.cpp matadd.o -o matadd$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart

nvcc -c pooling.cu -o pooling.o -x cu -Xcompiler -fPIC
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include pooling_wrapper.cpp pooling.o -o pooling$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart

# nvcc -c conv2d.cu -o conv2d.o -x cu -Xcompiler -fPIC
# g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include conv2d_wrapper.cpp conv2d.o -o conv2d$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart