# nvcc -c mat_mul.cu -o mat_mul.o -x cu -Xcompiler -fPIC
# g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include mat_mul_wrapper.cpp mat_mul.o -o mat_mul$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart

nvcc -c conv2d.cu -o conv2d.o -x cu -Xcompiler -fPIC
g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) -I/usr/local/cuda/include conv2d_wrapper.cpp conv2d.o -o conv2d$(python3-config --extension-suffix) -L/usr/local/cuda/lib64 -lcudart