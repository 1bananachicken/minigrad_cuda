#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include "conv2d.cuh"


__global__ void Conv2dForwardKernel(float *X, float *K, float *Y, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride)
{   
    int OH = (H - KH + 1) / stride;
    int OW = (W - KW + 1) / stride;

    int GEMM_M = C_out;
    int GEMM_K = C_in * KH * KW;
    int GEMM_N = N * OH * OW;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < GEMM_M && col < GEMM_N)
    {
        int oc = row;

        int n = col / (OH * OW);
        int j_res = col % (OH * OW);
        int oh = j_res / OW;
        int ow = j_res % OW;

        float accumulator = 0.0f;

        FORLOOP(k, GEMM_K)
        {
            int ic = k / (KH * KW);
            int k_res = k % (KH * KW);
            int kh = k_res / KW;
            int kw = k_res % KW;
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            float elem_k = K[IDX2C4D(oc, ic, kh, kw, C_in, KH, KW)];
            float elem_x = X[IDX2C4D(n, ic, ih, iw, C_in, H, W)];

            accumulator += elem_k * elem_x;
        }
        Y[IDX2C2D(row, col, GEMM_N)] = accumulator;
    }
}

__global__ void Conv2dBackwardKernelDX(float *K, float *Dout, float *DX, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride)
{
    int OH = (H - KH + 1) / stride;
    int OW = (W - KW + 1) / stride;

    int GEMM_M = C_in * KH * KW;
    int GEMM_K = C_out;
    int GEMM_N = N * OH * OW;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < GEMM_M && col < GEMM_N)
    {
        int ic = row / (KH * KW);
        int i_res = row % (KH * KW);
        int kh = i_res / KW;
        int kw = i_res % KW;
        int n = col / (OH * OW);
        int j_res = col % (OH * OW);
        int oh = j_res / OW;
        int ow = j_res % OW;
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;

        float accumulator = 0.0f;

        FORLOOP(k, GEMM_K)
        {
            int oc = k;
            float elem_k = K[IDX2C4D(oc, ic, kh, kw, C_in, KH, KW)];
            float elem_dout = Dout[IDX2C4D(n, ic, ih, iw, C_in, H, W)];
            accumulator += elem_k * elem_dout;

        }
        // 这里他妈一定不能写成+=，多线程冲突！！
        atomicAdd(&DX[IDX2C4D(n, ic, ih, iw, C_in, H, W)], accumulator);
    }
}

__global__ void Conv2dBackwardKernelDK(float *X, float *Dout, float *DK, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride)
{
    int OH = (H - KH + 1) / stride;
    int OW = (W - KW + 1) / stride;

    int GEMM_M = C_out;
    int GEMM_K = N * OH * OW;
    int GEMM_N = C_in * KH * KW;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < GEMM_M && col < GEMM_N)
    {
        int oc = row;
        int ic = col / (KH * KW);
        int j_res = col % (KH * KW);
        int kh = j_res / KW;
        int kw = j_res % KW;

        float accumulator = 0.0f;

        FORLOOP(k, GEMM_K)
        {
            int n = k / (OH * OW);
            int k_res = k % (OH * OW);
            int oh = k_res / OW;
            int ow = k_res % OW;
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            float elem_dout = Dout[IDX2C4D(n, oc, oh, ow, C_out, OH, OW)];
            float elem_x = X[IDX2C4D(n, ic, ih, iw, C_in, H, W)];

            accumulator += elem_dout * elem_x;
        }
        DK[IDX2C2D(row, col, GEMM_N)] = accumulator;
    }
}

// 数组索引别他妈溢出了
float* Conv2d(float *X, float *kernel, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O)
{   
    float *kernel_device, *X_device, *Y_device;
    float *Y_host = (float*)malloc(M * N_O * sizeof(float));
    checkCudaError(cudaMalloc(&kernel_device, M * K * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&X_device, N * C_in * H * W * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&Y_device, M * N_O * sizeof(float)), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(kernel_device, kernel, M * K * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(X_device, X, N * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((M + BLOCK_DIM - 1) / BLOCK_DIM, (N_O + BLOCK_DIM - 1) / BLOCK_DIM);
    Conv2dForwardKernel<<<grid_dim, block_dim>>>(X_device, kernel_device, Y_device, N, H, W, C_in, C_out, KH, KW, stride);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(Y_host, Y_device, M * N_O * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkCudaError(cudaFree(kernel_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(X_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(Y_device), __FILE__, __LINE__);
    return Y_host;
}

float* Conv2dBackwardDX(float *kernel, float *Dout, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O)
{
    float *kernel_device, *Dout_device, *DX_device;
    float *DX_host = (float*)malloc(N * C_in * H * W * sizeof(float));

    checkCudaError(cudaMalloc(&kernel_device, M * K * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&Dout_device, K * N_O * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&DX_device, N * C_in * H * W * sizeof(float)), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(kernel_device, kernel, M * K * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(Dout_device, Dout, K * N_O * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((M + BLOCK_DIM - 1) / BLOCK_DIM, (N_O + BLOCK_DIM - 1) / BLOCK_DIM);

    Conv2dBackwardKernelDX<<<grid_dim, block_dim>>>(kernel_device, Dout_device, DX_device, N, H, W, C_in, C_out, KH, KW, stride);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    checkCudaError(cudaMemcpy(DX_host, DX_device, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkCudaError(cudaFree(kernel_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(Dout_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(DX_device), __FILE__, __LINE__);

    return DX_host;
}

float* Conv2dBackwardDK(float *X, float *Dout, int N, int H, int W, int C_in, int C_out, int KH, int KW, int stride, int M, int K, int N_O)
{
    float *X_device, *Dout_device, *DK_device;
    float *DK_host = (float*)malloc(M * N_O * sizeof(float));

    checkCudaError(cudaMalloc(&X_device, N * C_in * H * W * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&Dout_device, M * K * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&DK_device, M * N_O * sizeof(float)), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(X_device, X, N * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(Dout_device, Dout, M * K * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((M + BLOCK_DIM - 1) / BLOCK_DIM, (N_O + BLOCK_DIM - 1) / BLOCK_DIM);

    Conv2dBackwardKernelDK<<<grid_dim, block_dim>>>(X_device, Dout_device, DK_device, N, H, W, C_in, C_out, KH, KW, stride);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    checkCudaError(cudaMemcpy(DK_host, DK_device, M * N_O * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkCudaError(cudaFree(X_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(Dout_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(DK_device), __FILE__, __LINE__);

    return DK_host;
}
