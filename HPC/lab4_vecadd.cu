#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

int main() {
    int N;
    std::cout << "Enter size of vectors: ";
    std::cin >> N;

    size_t size = N * sizeof(int);
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c = new int[N];

    std::cout << "Enter elements of vector A:\n";
    for (int i = 0; i < N; ++i) std::cin >> h_a[i];

    std::cout << "Enter elements of vector B:\n";
    for (int i = 0; i < N; ++i) std::cin >> h_b[i];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "Resultant Vector (A + B):\n";
    for (int i = 0; i < N; ++i) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;

    return 0;
}

