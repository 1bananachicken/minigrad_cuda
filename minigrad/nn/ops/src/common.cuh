#pragma once
#include <stdio.h>

#define BLOCK_DIM 32
#define IDX2C2D(i,j,w) ((i)*(w)+(j))
#define IDX2C4D(i,j,k,s,c,h,w) (((i)*(c)*(h)*(w))+((j)*(h)*(w))+((k)*(w))+(s))
#define FORLOOP(IDX, NLOOP) for(int IDX=0; IDX < NLOOP; ++IDX)

inline cudaError_t checkCudaError(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
                error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}