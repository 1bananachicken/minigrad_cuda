#include <cuda_runtime.h>
#include "matadd.cuh"

__global__ void matAdd2dKernel(float *A, float *B, float *C, int N, int C_out)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_B[BLOCK_DIM];

    if (row < N && col < C_out)
    {
        tile_B[threadIdx.y] = B[blockIdx.y * BLOCK_DIM + threadIdx.y];
        __syncthreads();
        C[IDX2C2D(row, col, C_out)] = A[IDX2C2D(row, col, C_out)] + tile_B[threadIdx.y];
    }
}

__global__ void matAdd4dKernel(float *A, float *B, float *C, int N, int C_out, int OH, int OW)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int ic = col / (OH * OW);
    int j_res = col % (OH * OW);
    int oh = j_res / OW;
    int ow = j_res % OW;

    if (row < N && col < C_out * OH * OW)
        C[IDX2C4D(row, ic, oh, ow, C_out, OH, OW)] = A[IDX2C4D(row, ic, oh, ow, C_out, OH, OW)] + B[ic];
}

float* matAdd2d(float *A, float *B, int N, int C_out)
{
    float *A_device, *B_device, *C_device;
    float *C = (float*)malloc(N * C_out * sizeof(float));

    checkCudaError(cudaMalloc(&A_device, N * C_out * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&B_device, C_out * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&C_device, N * C_out * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(A_device, A, N * C_out * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(B_device, B, C_out * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((N + block_dim.x - 1) / BLOCK_DIM, (C_out + block_dim.y - 1) / BLOCK_DIM);

    matAdd2dKernel<<<grid_dim, block_dim>>>(A_device, B_device, C_device, N, C_out);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    checkCudaError(cudaMemcpy(C, C_device, N * C_out * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    checkCudaError(cudaFree(A_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(B_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(C_device), __FILE__, __LINE__);

    return C;
}

float* matAdd4d(float *A, float *B, int N, int C_out, int OH, int OW)
{
    float *A_device, *B_device, *C_device;
    float *C = (float*)malloc(N * C_out * OH * OW * sizeof(float));

    checkCudaError(cudaMalloc(&A_device, N * C_out * OH * OW * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&B_device, C_out * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc(&C_device, N * C_out * OH * OW * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(A_device, A, N * C_out * OH * OW * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(B_device, B, C_out * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((N + block_dim.x - 1) / BLOCK_DIM, (C_out * OH * OW + block_dim.y - 1) / BLOCK_DIM);

    matAdd4dKernel<<<grid_dim, block_dim>>>(A_device, B_device, C_device, N, C_out, OH, OW);

    checkCudaError(cudaGetLastError(), __FILE__, __LINE__);
    checkCudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    checkCudaError(cudaMemcpy(C, C_device, N * C_out * OH * OW * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    checkCudaError(cudaFree(A_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(B_device), __FILE__, __LINE__);
    checkCudaError(cudaFree(C_device), __FILE__, __LINE__);

    return C;
}