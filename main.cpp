#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include "FFT.cpp"
int main(int argc, char* argv[]) {
    int num_threads = 1; // Default number of threads

    // Command line argument (Number of threads)
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }

    omp_set_num_threads(num_threads);

    int N;
    std::cout << "Enter the size of the sequences: ";
    std::cin >> N;

    std::vector<std::complex<double>> x(N), y(N);

    std::cout << "Enter the first sequence (space-separated): ";
    for (int i = 0; i < N; i++) {
        double real;
        std::cin >> real;
        x[i] = std::complex<double>(real, 0.0); // Assuming imaginary part is 0
    }

    std::cout << "Enter the second sequence (space-separated): ";
    for (int i = 0; i < N; i++) {
        double real;
        std::cin >> real;
        y[i] = std::complex<double>(real, 0.0); // Assuming imaginary part is 0
    }

    // Padding
    int M = 2*N;
    x.resize(M, 0);
    y.resize(M, 0);

    // Start timing
    double start_time = omp_get_wtime();

    // Calling FFT function
    std::vector<std::complex<double>> X = fft(x);
    std::vector<std::complex<double>> Y = fft(y);

    // Point-wise multiplication
    std::vector<std::complex<double>> result(M);
    #pragma omp parallel for
    for(int i = 0; i < M; i++) {
        result[i] = X[i] * Y[i];
    }

    // Calling iFFT
    std::vector<std::complex<double>> conv = ifft(result);

    // End timing
    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    std::cout << "Convolution result: ";
    for(int i = 0; i < M-1; i++) {
        std::cout << conv[i].real() << " ";
    }
    std::cout << std::endl;

    std::cout << "Time taken: " << time_taken*1000 << " milliseconds" << std::endl;
    std::cout << "Number of threads used: " << num_threads << std::endl;

    return 0;
}
