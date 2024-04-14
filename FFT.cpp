#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>


// FFT
std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return x;

    std::vector<std::complex<double>> even(N/2), odd(N/2);
    #pragma omp parallel for
    for (int i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }

    std::vector<std::complex<double>> q = fft(even);
    std::vector<std::complex<double>> r = fft(odd);

    std::vector<std::complex<double>> y(N);
    #pragma omp parallel for
    for (int k = 0; k < N/2; k++) {
        double theta = -2 * M_PI * k / N;
        std::complex<double> wk = std::polar(1.0, theta);
        y[k] = q[k] + wk * r[k];
        y[k + N/2] = q[k] - wk * r[k];
    }

    return y;
}

// IFFT
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x) {
    int N = x.size();
    #pragma omp parallel for
    for (auto& element : x) {
        element = std::conj(element);
    }
    std::vector<std::complex<double>> y = fft(x);
    #pragma omp parallel for
    for (auto& element : y) {
        element = std::conj(element) / static_cast<double>(N);
    }
    return y;
}
