#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pooling.cuh"
#include <stdio.h>

namespace py = pybind11;

py::tuple py_maxPool2d(py::array_t<float> X, int KH, int KW)
{
    auto X_buf = X.request();

    int N = X.shape(0);
    int C_in = X.shape(1);
    int H = X.shape(2);
    int W = X.shape(3);
    int OH = H / KH;
    int OW = W / KW;

    float *X_ptr = static_cast<float*>(X_buf.ptr);

    auto [B_ptr, indices_ptr] = pooling2d(X_ptr, N, C_in, H, W, KH, KW, MAXPOOL);

    py::array_t<float> output(
        {N, C_in, OH, OW},
        {sizeof(float) * C_in * OH * OW, sizeof(float) * OH * OW, sizeof(float) * OW, sizeof(float)},
        B_ptr
    );

    py::array_t<int> indices(
        {N, C_in, OH, OW},
        {sizeof(int) * C_in * OH * OW, sizeof(int) * OH * OW, sizeof(int) * OW, sizeof(int)},
        indices_ptr
    );

    free(B_ptr);
    free(indices_ptr);
    return py::make_tuple(output, indices);
}

py::array_t<float> py_avgPool2d(py::array_t<float> X, int KH, int KW)
{
    auto X_buf = X.request();

    int N = X.shape(0);
    int C_in = X.shape(1);
    int H = X.shape(2);
    int W = X.shape(3);
    int OH = H / KH;
    int OW = W / KW;

    float *X_ptr = static_cast<float*>(X_buf.ptr);

    auto [B_ptr, mask] = pooling2d(X_ptr, N, C_in, H, W, KH, KW, AVGPOOL);

    py::array_t<float> result(
        {N, C_in, OH, OW},
        {sizeof(float) * C_in * OH * OW, sizeof(float) * OH * OW, sizeof(float) * OW, sizeof(float)},
        B_ptr
    );

    free(B_ptr);

    return result;
}

py::array_t<float> py_maxPool2dBackward(py::array_t<float> Dout, py::array_t<int> indices, int KH, int KW)
{
    auto Dout_buf = Dout.request();
    auto indices_buf = indices.request();

    int N = Dout.shape(0);
    int C_in = Dout.shape(1);
    int OH = Dout.shape(2);
    int OW = Dout.shape(3);
    int H = OH * KH;
    int W = OW * KW;

    float *Dout_ptr = static_cast<float*>(Dout_buf.ptr);
    int *indices_ptr = static_cast<int*>(indices_buf.ptr);
    float *DX = (float*)malloc(N * C_in * H * W * sizeof(float));

    FORLOOP(i, N * C_in * H * W)
    {
        DX[i] = 0;
    }
    FORLOOP(i, N * C_in * OH * OW)
    {
        DX[indices_ptr[i]] = Dout_ptr[i];
    }

    py::array_t<float> dx(
        {N, C_in, H, W},
        {sizeof(float) * C_in * H * W, sizeof(float) * H * W, sizeof(float) * W, sizeof(float)},
        DX
    );

    free(DX);
    return dx;
}

py::array_t<float> py_avgPool2dBackward(py::array_t<float> Dout, int KH, int KW)
{
    auto Dout_buf = Dout.request();

    int N = Dout.shape(0);
    int C_in = Dout.shape(1);
    int OH = Dout.shape(2);
    int OW = Dout.shape(3);
    int H = OH * KH;
    int W = OW * KW;

    float *Dout_ptr = static_cast<float*>(Dout_buf.ptr);
    float *DX_ptr = avgPool2dBackward(Dout_ptr, N, C_in, H, W, KH, KW);

    py::array_t<float> result(
        {N, C_in, H, W},
        {sizeof(float) * C_in * H * W, sizeof(float) * H * W, sizeof(float) * W, sizeof(float)},
        DX_ptr
    );

    free(DX_ptr);

    return result;
}

PYBIND11_MODULE(pooling, m)
{
    m.def("maxPool2d", &py_maxPool2d, "maxPool2d",
        py::arg("X"), py::arg("KH"), py::arg("KW"));
    m.def("avgPool2d", &py_avgPool2d, "avgPool2d",
        py::arg("X"), py::arg("KH"), py::arg("KW"));
    m.def("maxPool2dBackward", &py_maxPool2dBackward, "maxPool2d backward",
        py::arg("Dout"), py::arg("indices"), py::arg("KH"), py::arg("KW"));
    m.def("avgPool2dBackward", &py_avgPool2dBackward, "avgPool2d backward",
        py::arg("Dout"), py::arg("KH"), py::arg("KW"));
}