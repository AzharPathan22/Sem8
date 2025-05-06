#include <iostream>
#include <omp.h>
#include <chrono>
#include <tuple>

using namespace std;
using namespace chrono;

// Simple static array for demonstration
const int SIZE = 20;
int inputArr[SIZE] = {34, 7, 23, 32, 5, 62, 78, 12, 17, 8, 45, 90, 3, 15, 28, 67, 49, 2, 99, 1};

// Sequential Bubble Sort
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

// Parallel Bubble Sort
void parallelBubbleSort(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        int start = i % 2;
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
    }
}

// Merge for Merge Sort
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int* L = new int[n1];
    int* R = new int[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallelMergeSort(int arr[], int l, int r, int depth = 0) {
    if (l < r) {
        int m = l + (r - l) / 2;
        if (depth <= 4) {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, l, m, depth + 1);
                #pragma omp section
                parallelMergeSort(arr, m + 1, r, depth + 1);
            }
        } else {
            mergeSort(arr, l, m);
            mergeSort(arr, m + 1, r);
        }
        merge(arr, l, m, r);
    }
}

// Efficiency calculation
tuple<double, double, double> calcEfficiency(double seqTime, double parTime, int threads) {
    if (parTime == 0) parTime = 1e-6;
    double speedup = seqTime / parTime;
    double cost = parTime * threads;
    double efficiency = speedup / threads;
    return make_tuple(speedup, cost, efficiency);
}

// Utility to print arrays
void printArray(const string& label, int arr[], int size) {
    cout << label;
    for (int i = 0; i < size; i++) cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int numThreads = 8;

    int arr1[SIZE], arr2[SIZE], arr3[SIZE], arr4[SIZE];
    copy(inputArr, inputArr + SIZE, arr1);
    copy(inputArr, inputArr + SIZE, arr2);
    copy(inputArr, inputArr + SIZE, arr3);
    copy(inputArr, inputArr + SIZE, arr4);

    omp_set_num_threads(numThreads);

    auto start = high_resolution_clock::now();
    bubbleSort(arr1, SIZE);
    auto end = high_resolution_clock::now();
    double seqBubbleTime = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    parallelBubbleSort(arr2, SIZE);
    end = high_resolution_clock::now();
    double parBubbleTime = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    mergeSort(arr3, 0, SIZE - 1);
    end = high_resolution_clock::now();
    double seqMergeTime = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    parallelMergeSort(arr4, 0, SIZE - 1);
    end = high_resolution_clock::now();
    double parMergeTime = duration_cast<microseconds>(end - start).count();

    // Print arrays
    printArray("Sorted Bubble Seq:  ", arr1, SIZE);
    printArray("Sorted Bubble Para: ", arr2, SIZE);
    printArray("Sorted Merge Seq:   ", arr3, SIZE);
    printArray("Sorted Merge Para:  ", arr4, SIZE);

    // Output times and efficiency
    double s, c, e;
    tie(s, c, e) = calcEfficiency(seqBubbleTime, parBubbleTime, numThreads);
    double sm, cm, em;
    tie(sm, cm, em) = calcEfficiency(seqMergeTime, parMergeTime, numThreads);

    cout << "\n--- Results (Time in ms) ---\n";
    cout << "Sequential Bubble Sort Time: " << seqBubbleTime << " ms\n";
    cout << "Parallel Bubble Sort Time:   " << parBubbleTime << " ms\n";
    cout << "Sequential Merge Sort Time:  " << seqMergeTime << " ms\n";
    cout << "Parallel Merge Sort Time:    " << parMergeTime << " ms\n";

    cout << "\n--- Cost & Efficiency ---\n";
    cout << "Bubble Sort - Speedup: " << s << ", Cost: " << c << ", Efficiency: " << e << "\n";
    cout << "Merge Sort  - Speedup: " << sm << ", Cost: " << cm << ", Efficiency: " << em << "\n";

    return 0;
}

