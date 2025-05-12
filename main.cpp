#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <vector>

bool is_avx2_supported() {
    return __builtin_cpu_supports("avx2");
}

void fill_random(float *array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);
    
    for (int i = 0; i < size; ++i) {
        array[i] = dist(gen);
    }
}

void multiply_avx2(const float *array1, const float *array2, float *result, int size) {
    int i = 0;
    
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&array1[i]);
        __m256 vec2 = _mm256_loadu_ps(&array2[i]);
        __m256 vec_result = _mm256_mul_ps(vec1, vec2);
        _mm256_storeu_ps(&result[i], vec_result);
    }
    
    for (; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

void multiply_optimized(const float *array1, const float *array2, float *result, int size) {
    int i = 0;
    
    for (; i <= size - 8; i += 8) {
        result[i] = array1[i] * array2[i];
        result[i + 1] = array1[i + 1] * array2[i + 1];
        result[i + 2] = array1[i + 2] * array2[i + 2];
        result[i + 3] = array1[i + 3] * array2[i + 3];
        result[i + 4] = array1[i + 4] * array2[i + 4];
        result[i + 5] = array1[i + 5] * array2[i + 5];
        result[i + 6] = array1[i + 6] * array2[i + 6];
        result[i + 7] = array1[i + 7] * array2[i + 7];
    }
    
    for (; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

void multiply_scalar(const float *array1, const float *array2, float *result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

int main_run() {
    const int N = 1024 * 1024 * 256;
    std::vector<float> array1(N), array2(N), result_avx2(N), result_optimized(N), result_scalar(N);
    
    fill_random(array1.data(), N);
    fill_random(array2.data(), N);
    
    if (!is_avx2_supported()) {
        std::cerr << "AVX2 is not supported on this CPU!\n";
        return 1;
    }
    
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    multiply_avx2(array1.data(), array2.data(), result_avx2.data(), N);
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_avx2 = end_avx2 - start_avx2;
    
    auto start_optimized = std::chrono::high_resolution_clock::now();
    multiply_optimized(array1.data(), array2.data(), result_optimized.data(), N);
    auto end_optimized = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_optimized = end_optimized - start_optimized;
    
    auto start_scalar = std::chrono::high_resolution_clock::now();
    multiply_scalar(array1.data(), array2.data(), result_scalar.data(), N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_scalar = end_scalar - start_scalar;
    
    bool results_match = true;
    for (int i = 0; i < N; ++i) {
        if (result_avx2[i] != result_scalar[i] || result_optimized[i] != result_scalar[i]) {
            results_match = false;
            break;
        }
    }
    
    std::cout << "Results match: " << (results_match ? "Yes" : "No") << "\n";
    std::cout << "Time taken (AVX2): " << elapsed_avx2.count() << " seconds\n";
    std::cout << "Time taken (Optimized): " << elapsed_optimized.count() << " seconds\n";
    std::cout << "Time taken (Scalar): " << elapsed_scalar.count() << " seconds\n";
    
    return 0;
}

int main() {
    for (int i = 0; i < 10; i++) {
        if (main_run()) return 1;
    }
    return 0;
}