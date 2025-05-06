#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for Parallel Reduction
__global__ void reductionKernel(int *d_data, int *d_result, int size, int op) {
    extern __shared__ int s_data[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        s_data[tid] = d_data[idx];
    }
    else {
        s_data[tid] = (op == 0) ? INT_MAX : (op == 1) ? INT_MIN : 0; // Handle boundary conditions
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (op == 0) s_data[tid] = min(s_data[tid], s_data[tid + s]); // Min operation
            else if (op == 1) s_data[tid] = max(s_data[tid], s_data[tid + s]); // Max operation
            else s_data[tid] += s_data[tid + s]; // Sum operation
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        d_result[blockIdx.x] = s_data[0];
    }
}

// Function to calculate final reduction
int reduce(int *d_data, int size, int op) {
    int *d_result;
    int block_size = 256; // Number of threads per block
    int grid_size = (size + block_size - 1) / block_size;
    int result = 0;

    // Allocate memory for the result
    cudaMalloc(&d_result, grid_size * sizeof(int));

    // Launch the reduction kernel
    reductionKernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_data, d_result, size, op);

    // Copy back the results
    int *h_result = new int[grid_size];
    cudaMemcpy(h_result, d_result, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction in CPU
    result = h_result[0];
    for (int i = 1; i < grid_size; i++) {
        if (op == 0) result = min(result, h_result[i]);
        else if (op == 1) result = max(result, h_result[i]);
        else result += h_result[i];
    }

    // Free the allocated memory
    delete[] h_result;
    cudaFree(d_result);

    return result;
}

int main() {
    // Set up size of the data
    int size = 1024; // Size of the array
    int *h_data = new int[size];

    // Initialize data
    srand(time(0));
    for (int i = 0; i < size; i++) {
        h_data[i] = rand() % 1000; // Random numbers between 0 and 999
    }

    // Allocate device memory
    int *d_data;
    cudaMalloc(&d_data, size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Perform Parallel Reduction for Sum
    int sum = reduce(d_data, size, 2); // 2 for Sum
    cout << "Sum: " << sum << endl;

    // Perform Parallel Reduction for Min
    int min_val = reduce(d_data, size, 0); // 0 for Min
    cout << "Min: " << min_val << endl;

    // Perform Parallel Reduction for Max
    int max_val = reduce(d_data, size, 1); // 1 for Max
    cout << "Max: " << max_val << endl;

    // Compute Average
    float avg = sum / (float)size;
    cout << "Average: " << avg << endl;

    // Free device memory
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}

