#include <stdio.h>
#include <cuda_runtime_api.h>
#include "matmul.cuh"
#include <stdlib.h>

__global__ void matMulKernel(float *A, float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tile_B[BLOCK_DIM][BLOCK_DIM];

    float accumulator = 0.0f;
    int num_blocks = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    
    FORLOOP(b, num_blocks)
    {
        if (row < M && threadIdx.y + b * BLOCK_DIM < K)
            tile_A[threadIdx.x][threadIdx.y] = A[row * K + threadIdx.y + b * BLOCK_DIM];
        else 
            tile_A[threadIdx.x][threadIdx.y] = 0.0f;

        if (threadIdx.x + b * BLOCK_DIM < K && col < N)
            tile_B[threadIdx.x][threadIdx.y] = B[(threadIdx.x + b * BLOCK_DIM) * N + col];
        else
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        FORLOOP(i, BLOCK_DIM)
        {
            accumulator += tile_A[threadIdx.x][i] * tile_B[i][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[IDX2C2D(row, col, N)] = accumulator;
}

__global__ void matMulKernelATranspose(float *A, float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tile_B[BLOCK_DIM][BLOCK_DIM];

    float accumulator = 0.0f;
    int num_blocks = (K + BLOCK_DIM - 1) / BLOCK_DIM;

    FORLOOP(b, num_blocks)
    {
        if (row < M && threadIdx.y + b * BLOCK_DIM < K)
            tile_A[threadIdx.x][threadIdx.y] = A[(threadIdx.y + b * BLOCK_DIM) * M + row];
        else
            tile_A[threadIdx.x][threadIdx.y] = 0.0f;

        if (threadIdx.x + b * BLOCK_DIM < K && col < N)
            tile_B[threadIdx.x][threadIdx.y] = B[(threadIdx.x + b * BLOCK_DIM) * N + col];
        else
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        FORLOOP(i, BLOCK_DIM)
        {
            accumulator += tile_A[threadIdx.x][i] * tile_B[i][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[IDX2C2D(row, col, N)] = accumulator;
}

__global__ void matMulKernelBTranspose(float *A, float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float tile_B[BLOCK_DIM][BLOCK_DIM];

    float accumulator = 0.0f;
    int num_blocks = (K + BLOCK_DIM - 1) / BLOCK_DIM;

    FORLOOP(b, num_blocks)
    {
        if (row < M && threadIdx.y + b * BLOCK_DIM < K)
            tile_A[threadIdx.x][threadIdx.y] = A[row * K + threadIdx.y + b * BLOCK_DIM];
        else
            tile_A[threadIdx.x][threadIdx.y] = 0.0f;

        if (threadIdx.x + b * BLOCK_DIM < K && col < N)
            tile_B[threadIdx.x][threadIdx.y] = B[col * K + threadIdx.x + b * BLOCK_DIM];
        else
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        FORLOOP(i, BLOCK_DIM)
        {
            accumulator += tile_A[threadIdx.x][i] * tile_B[i][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[IDX2C2D(row, col, N)] = accumulator;
}

float* matMul(float *A, float *B, int M, int K, int N, bool AT, bool BT)
{   
    float *A_device, *B_device, *C_device;
    float *C_host = (float*)malloc(M * N * sizeof(float));

    cudaMalloc(&A_device, M * K * sizeof(float));
    cudaMalloc(&B_device, K * N * sizeof(float));
    cudaMalloc(&C_device, M * N * sizeof(float));

    cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid_dim((M + block_dim.x - 1) / BLOCK_DIM, (N + block_dim.y - 1) / BLOCK_DIM, 1);
    if (AT)
        matMulKernelATranspose<<<grid_dim, block_dim>>>(A_device, B_device, C_device, M, K, N);
    else if (BT)
        matMulKernelBTranspose<<<grid_dim, block_dim>>>(A_device, B_device, C_device, M, K, N);
    else
        matMulKernel<<<grid_dim, block_dim>>>(A_device, B_device, C_device, M, K, N);

    cudaDeviceSynchronize();
    
    cudaMemcpy(C_host, C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    
    return C_host;
}