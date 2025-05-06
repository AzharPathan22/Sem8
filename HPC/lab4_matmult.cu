#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulKernel(int* A, int* B, int* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int sum = 0;
        for (int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}

int main() {
    int m, n, k;
    std::cout << "Enter dimensions of Matrix A (m x n): ";
    std::cin >> m >> n;
    std::cout << "Enter dimensions of Matrix B (n x k): ";
    std::cin >> n >> k;

    size_t sizeA = m * n * sizeof(int);
    size_t sizeB = n * k * sizeof(int);
    size_t sizeC = m * k * sizeof(int);

    int* h_A = new int[m * n];
    int* h_B = new int[n * k];
    int* h_C = new int[m * k];

    std::cout << "Enter elements of Matrix A (" << m << "x" << n << "):\n";
    for (int i = 0; i < m * n; ++i) std::cin >> h_A[i];

    std::cout << "Enter elements of Matrix B (" << n << "x" << k << "):\n";
    for (int i = 0; i < n * k; ++i) std::cin >> h_B[i];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((k + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "Resultant Matrix (A x B):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j)
            std::cout << h_C[i * k + j] << " ";
        std::cout << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}

