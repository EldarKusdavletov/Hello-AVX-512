#include <iostream>
#include <immintrin.h> // AVX2 intrinsics
#include <chrono>      // For timing
#include <random>      // For random number generation
#include <vector>      // For dynamic memory allocation

// Function to check AVX2 support
bool is_avx2_supported() {
    return __builtin_cpu_supports("avx2");
}

// Function to fill an array with random float values
void fill_random(float* array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);
    
    for (int i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }
}

// AVX2 implementation of array multiplication
void multiply_avx2(const float* array1, const float* array2, float* result, int size) {
    int i = 0;
    // Process 8 elements at a time
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&array1[i]);
        __m256 vec2 = _mm256_loadu_ps(&array2[i]);
        __m256 vec_result = _mm256_mul_ps(vec1, vec2);
        _mm256_storeu_ps(&result[i], vec_result);
    }
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

// Scalar (non-AVX2) implementation of array multiplication
void multiply_scalar(const float* array1, const float* array2, float* result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

int main() {
    const int N = 1024 * 1024 * 1024; // 1 billion elements (multiple of 8)
    std::vector<float> array1(N), array2(N), result_avx2(N), result_scalar(N);
    
    // Fill arrays with random float values
    fill_random(array1.data(), N);
    fill_random(array2.data(), N);
    
    // Check AVX2 support
    if (!is_avx2_supported()) {
        std::cerr << "AVX2 is not supported on this CPU!\n";
        return 1;
    }
    
    // Measure AVX2 performance
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    multiply_avx2(array1.data(), array2.data(), result_avx2.data(), N);
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_avx2 = end_avx2 - start_avx2;
    
    // Measure scalar performance
    auto start_scalar = std::chrono::high_resolution_clock::now();
    multiply_scalar(array1.data(), array2.data(), result_scalar.data(), N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_scalar = end_scalar - start_scalar;
    
    // Verify results (optional)
    bool results_match = true;
    for (int i = 0; i < N; ++i) {
        if (result_avx2[i] != result_scalar[i]) {
            results_match = false;
            break;
        }
    }
    
    // Print results and performance
    std::cout << "First 5 results (AVX2):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << result_avx2[i] << " ";
    }
    std::cout << "\nFirst 5 results (Scalar):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << result_scalar[i] << " ";
    }
    std::cout << "\n\nResults match: " << (results_match ? "Yes" : "No") << "\n";
    std::cout << "Time taken (AVX2): " << elapsed_avx2.count() << " seconds\n";
    std::cout << "Time taken (Scalar): " << elapsed_scalar.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed_scalar.count() / elapsed_avx2.count() << "x\n";
    
    return 0;
}