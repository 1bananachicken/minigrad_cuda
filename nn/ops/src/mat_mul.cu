#include <stdio.h>
#include <cuda_runtime_api.h>
#include "mat_mul.cuh"
#include <stdlib.h>

__global__ void matMulKernel(float *A, float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_A[32][32];
    __shared__ float tile_B[32][32];

    float accumulator = 0.0f;
    int num_blocks = (K + blockDim.x - 1) / blockDim.x;
    
    for(int b = 0; b < num_blocks; ++b)
    {
        if (row < M && threadIdx.y + b * blockDim.x < K)
            tile_A[threadIdx.x][threadIdx.y] = A[row * K + threadIdx.y + b * blockDim.x];
        else 
            tile_A[threadIdx.x][threadIdx.y] = 0.0f;

        if (threadIdx.x + b * blockDim.x < K && col < N)
            tile_B[threadIdx.x][threadIdx.y] = B[(threadIdx.x + b * blockDim.x) * N + col];
        else
            tile_B[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i)
        {
            accumulator += tile_A[threadIdx.x][i] * tile_B[i][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = accumulator;
}

float* matMul(float *A, float *B, int M, int K, int N)
{   
    float *A_device, *B_device, *C_device;
    float *C_host = (float*)malloc(M * N * sizeof(float));

    cudaMalloc(&A_device, M * K * sizeof(float));
    cudaMalloc(&B_device, K * N * sizeof(float));
    cudaMalloc(&C_device, M * N * sizeof(float));

    cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(32, 32, 1);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y, 1);

    matMulKernel<<<grid_dim, block_dim>>>(A_device, B_device, C_device, M, K, N);

    cudaDeviceSynchronize();
    
    cudaMemcpy(C_host, C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    
    return C_host;
}

int main()
{
    int M = 33, K = 33, N = 33;
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    float *A_device, *B_device, *C_device;
    
    cudaMalloc(&A_device, M * K * sizeof(float));
    cudaMalloc(&B_device, K * N * sizeof(float));
    cudaMalloc(&C_device, M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i)
        A[i] = (float)rand() / RAND_MAX;
    
    for (int i = 0; i < K * N; ++i)
        B[i] = (float)rand() / RAND_MAX;

    cudaMemcpy(A_device, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block_dim(32, 32);
    dim3 grid_dim((M + block_dim.x - 1) / block_dim.x, (N + block_dim.y - 1) / block_dim.y);

    matMulKernel<<<grid_dim, block_dim>>>(A_device, B_device, C_device, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, C_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M*N; ++i)
        printf("%f\n", C[i]);
}