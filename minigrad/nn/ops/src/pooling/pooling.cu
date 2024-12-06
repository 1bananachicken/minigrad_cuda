#include <cuda_runtime.h>
#include <float.h>
#include <tuple>
#include "pooling.cuh"

__global__ void maxPool2dKernel(float *A, float *B, int *indices, int N, int C_in, int H, int W, int KH, int KW)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int OH = H / KH;
    int OW = W / KW;
    if (row < N && col < C_in * OH * OW)
    {
        int ic = col / (OH * OW);
        int j_res = col % (OH * OW);
        int ih = j_res / OW;
        int iw = j_res % OW;
        // 谁他妈知道为啥这里不能写成 -1e30？？
        float max_elem = -1.0 * 1e30;
        int max_idx = 0;
        FORLOOP(offset_h, KH)
        {
            FORLOOP(offset_w, KW)
            {
                int elem_A_idx = IDX2C4D(row, ic, ih*KH+offset_h, iw*KW+offset_w, C_in, H, W);
                float elem_A = A[elem_A_idx];
                if (elem_A > max_elem)
                {
                    max_elem = elem_A;
                    max_idx = elem_A_idx;
                }
            }
        }
        B[IDX2C2D(row, col, C_in * OH * OW)] = max_elem;
        indices[IDX2C2D(row, col, C_in * OH * OW)] = max_idx;
    }
}

__global__ void avgPool2dKernel(float *A, float *B, int N, int C_in, int H, int W, int KH, int KW)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int OH = H / KH;
    int OW = W / KW;
    if (row < N && col < C_in * OH * OW)
    {
        int ic = col / (OH * OW);
        int j_res = col % (OH * OW);
        int ih = j_res / OW;
        int iw = j_res % OW;
        float avg = 0.0f;
        
        FORLOOP(offset_h, KH)
        {
            FORLOOP(offset_w, KW)
            {
                float elem_A = A[IDX2C4D(row, ic, ih*KH+offset_h, iw*KW+offset_w, C_in, H, W)];
                avg += elem_A;
            }
        }
        B[IDX2C2D(row, col, C_in * OH * OW)] = avg / (KH * KW);
    }
}

__global__ void avgPool2dBackwardKernel(float *Dout, float *DX, int N, int C_in, int H, int W, int KH, int KW)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int OH = H / KH;
    int OW = W / KW;
    if (row < N && col < C_in * OH * OW)
    {
        int ic = col / (OH * OW);
        int j_res = col % (OH * OW);
        int ih = j_res / OW;
        int iw = j_res % OW;
        float avg_dout = Dout[IDX2C2D(row, col, C_in * OH * OW)] / (KH * KW);
        
        FORLOOP(offset_h, KH)
        {
            FORLOOP(offset_w, KW)
            {
                DX[IDX2C4D(row, ic, ih*KH+offset_h, iw*KW+offset_w, C_in, H, W)] = avg_dout;
            }
        }
    }
}

std::tuple<float*, int*> pooling2d(float *A, int N, int C_in, int H, int W, int KH, int KW, poolingType ptype)
{
    float *A_device, *B_device;
    int *indices, *indices_device;
    int OH = H / KH;
    int OW = W / KW;
    float *B = (float*)malloc(N * C_in * OH * OW * sizeof(float));

    checkCudaError(cudaMalloc(&A_device, N * C_in * H * W * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&B_device, N * C_in * OH * OW  * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(A_device, A, N * C_in * H * W  * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(B_device, B, N * C_in * OH * OW  * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((N + BLOCK_DIM - 1) / BLOCK_DIM, (C_in * OH * OW  + BLOCK_DIM- 1) / BLOCK_DIM);

    if (ptype == 0)
    {
        indices = (int*)malloc(N * C_in * OH * OW * sizeof(int));
        FORLOOP(i, N*C_in*OH*OW)
        {
            indices[i] = 0;
        }
        checkCudaError(cudaMalloc(&indices_device, N * C_in * OH * OW * sizeof(int)), __FILE__, __LINE__);
        checkCudaError(cudaMemcpy(indices_device, indices, N * C_in * OH * OW  * sizeof(int), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        maxPool2dKernel<<<grid_dim, block_dim>>>(A_device, B_device, indices_device, N, C_in, H ,W, KH, KW);

        checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
        checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

        checkCudaError(cudaMemcpy(indices, indices_device, N * C_in * OH * OW* sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        checkCudaError(cudaFree(indices_device), __FILE__, __LINE__);
    }   
    else
    {
        avgPool2dKernel<<<grid_dim, block_dim>>>(A_device, B_device, N, C_in, H ,W, KH, KW);
        checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
        checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }

    checkCudaError(cudaMemcpy(B, B_device, N * C_in * OH * OW * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    checkCudaError(cudaFree(A_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(B_device), __FILE__, __LINE__);

    return std::make_tuple(B, indices);
}

float* avgPool2dBackward(float *Dout, int N, int C_in, int H, int W, int KH, int KW)
{
    float *DX_device, *Dout_device;
    int OH = H / KH;
    int OW = W / KW;
    float *DX = (float*)malloc(N * C_in * H * W * sizeof(float));

    checkCudaError(cudaMalloc(&DX_device, N * C_in * H * W * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&Dout_device, N * C_in * OH * OW  * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(DX_device, DX, N * C_in * H * W  * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(Dout_device, Dout, N * C_in * OH * OW  * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((N + BLOCK_DIM - 1) / BLOCK_DIM, (C_in * OH * OW  + BLOCK_DIM- 1) / BLOCK_DIM);

    avgPool2dBackwardKernel<<<grid_dim, block_dim>>>(Dout_device, DX_device, N, C_in, H, W, KH, KW);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(DX, DX_device, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    checkCudaError(cudaFree(DX_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(Dout_device), __FILE__, __LINE__);

    return DX;
}