#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "conv2dv2.cuh"
#include <stdio.h>

namespace py = pybind11;

py::array_t<float> py_conv2dv2(py::array_t<float> X, py::array_t<float> kernel, int stride)
{
    auto X_buf = X.request();
    auto kernel_buf = kernel.request();\

    int N = X.shape(0);
    int C_in = X.shape(1);
    int H = X.shape(2);
    int W = X.shape(3);
    int C_out = kernel.shape(0);
    int KH = kernel.shape(2);
    int KW = kernel.shape(3);
    int OH = (H - KH) / stride + 1;
    int OW = (W - KW) / stride + 1;
    int M = C_out;
    int K = C_in * KH * KW;
    int N_O = N * OH * OW;

    float *X_ptr = static_cast<float*>(X_buf.ptr);
    float *kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    float *C_ptr = Conv2d(X_ptr, kernel_ptr, N, H, W, C_in, C_out, KH, KW, stride, M, K, N_O);

    py::array_t<float> result(
        {N, C_out, OH, OW},
        {sizeof(float) * C_out * OH * OW, sizeof(float) * OH * OW, sizeof(float) * OW, sizeof(float)},
        C_ptr
    );

    free(C_ptr);

    return result;
}

py::tuple py_conv2d_backward(py::array_t<float> X, py::array_t<float> kernel, py::array_t<float> Dout, int stride)
{
    auto X_buf = X.request();
    auto kernel_buf = kernel.request();
    auto Dout_buf = Dout.request();

    float *X_ptr = static_cast<float*>(X_buf.ptr);
    float *kernel_ptr = static_cast<float*>(kernel_buf.ptr);
    float *Dout_ptr = static_cast<float*>(Dout_buf.ptr);

    int N = X.shape(0);
    int C_in = X.shape(1);
    int H = X.shape(2);
    int W = X.shape(3);
    int C_out = kernel.shape(0);
    int KH = kernel.shape(2);
    int KW = kernel.shape(3);
    int OH = Dout.shape(2);
    int OW = Dout.shape(3);

    int M = C_in * KH * KW;
    int K = C_out;
    int N_O = N * OH * OW;

    float *DX_ptr = Conv2dBackwardDX(kernel_ptr, Dout_ptr, N, H, W, C_in, C_out, KH, KW, stride, M, K, N_O);
    float *DK_ptr = Conv2dBackwardDK(X_ptr, Dout_ptr, N, H, W, C_in, C_out, KH, KW, stride, K, N_O, M);

    py::array_t<float> DX(
        {N, C_in, H, W},
        {sizeof(float) * C_in * H * W, sizeof(float) * H * W, sizeof(float) * W, sizeof(float)},
        DX_ptr
    );

    py::array_t<float> DK(
        {C_out, C_in, KH, KW},
        {sizeof(float) * C_in * KH * KW, sizeof(float) * KH * KW, sizeof(float) * KW, sizeof(float)},
        DK_ptr
    );

    free(DX_ptr);
    free(DK_ptr);

    return py::make_tuple(DX, DK);
}

PYBIND11_MODULE(conv2dv2, m)
{
    m.def("conv2dv2", &py_conv2dv2, "Conv2d forward",
        py::arg("X"), py::arg("kernel"), py::arg("stride"));
    m.def("conv2d_backward", &py_conv2d_backward, "Conv2d backward",
        py::arg("x"), py::arg("kernel"), py::arg("dout"), py::arg("stride"));
}
