#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <chrono>
#include <random>

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

void multiply_scalar(const float *array1, const float *array2, float *result, int size) {
    for (int i = 0; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

void multiply_unrolled(const float *array1, const float *array2, float *result, int size) {
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

void multiply_avx2(const float *array1, const float *array2, float *result, int size) {
    int i = 0;
    
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_load_ps(&array1[i]);  // Aligned load
        __m256 vec2 = _mm256_load_ps(&array2[i]);  // Aligned load
        __m256 vec_result = _mm256_mul_ps(vec1, vec2);
        _mm256_store_ps(&result[i], vec_result);  // Aligned store
    }
    
    for (; i < size; ++i) {
        result[i] = array1[i] * array2[i];
    }
}

inline double timer(const float *array1, const float *array2, float *result, int size, void (*func)(const float *, const float *, float *, int), int iterations) {
    std::ofstream file("misc/timing.csv", std::ios::app);
    double total_time = 0;
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(array1, array2, result, size);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
        
        if (file.is_open()) {
            file << elapsed.count() << ',';
        }
    }
    
    if (file.is_open()) {
        file << '\n';
    }
    
    return total_time / iterations;
}

int main(int argc, char *argv[]) {
    if (!is_avx2_supported()) {
        std::cerr << "AVX2 is not supported on this CPU!\n";
        return 1;
    }
    
    std::ofstream file("misc/timing.csv", std::ios::trunc);
    file.close();
    
    const int N = 1024 * 1024 * 256;
    int iterations = 1;
    if (argc == 2) iterations = atoi(argv[1]);
    
    float *array1 = (float *) _mm_malloc(N * sizeof(float), 32);
    float *array2 = (float *) _mm_malloc(N * sizeof(float), 32);
    float *result_avx2 = (float *) _mm_malloc(N * sizeof(float), 32);
    float *result_unrolled = (float *) _mm_malloc(N * sizeof(float), 32);
    float *result_scalar = (float *) _mm_malloc(N * sizeof(float), 32);
    
    fill_random(array1, N);
    fill_random(array2, N);
    
    std::cout << "Time taken (Scalar): " << timer(array1, array2, result_scalar, N, multiply_scalar, iterations) << " seconds\n";
    std::cout << "Time taken (Unrolled): " << timer(array1, array2, result_unrolled, N, multiply_unrolled, iterations) << " seconds\n";
    std::cout << "Time taken (AVX2): " << timer(array1, array2, result_avx2, N, multiply_avx2, iterations) << " seconds\n";
    
    bool results_match = true;
    for (int i = 0; i < N; ++i) {
        if (result_scalar[i] != result_unrolled[i] || result_scalar[i] != result_avx2[i]) {
            results_match = false;
            break;
        }
    }
    
    std::cout << "Results match: " << (results_match ? "Yes" : "No") << "\n";
    
    _mm_free(array1);
    _mm_free(array2);
    _mm_free(result_avx2);
    _mm_free(result_unrolled);
    _mm_free(result_scalar);
    
    return 0;
}
