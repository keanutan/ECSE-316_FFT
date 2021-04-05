import numpy as np
import time


class DFT:

    @staticmethod
    def DFT_Slow_One_Dimension(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        dft = np.zeros(N, dtype=complex)

        # Following the lab instructions.
        for k in range(N):
            for n in range(N):
                dft[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        return dft

    @staticmethod
    def DFT_Slow_One_Dimension_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        idft = np.zeros(N, dtype=complex)

        # Following the lab instructions.
        for n in range(N):
            for k in range(N):
                idft[n] += x[k] * np.exp(2j * np.pi * k * n / N) / N
        return idft

    @staticmethod
    def FFT_Cooley_Tukey(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]

        # Using Cooley-Tukey FFT algorithm.
        if N % 2 != 0:
            raise ValueError("Size of x must be a power of 2!")
        if N <= 2:
            return DFT.DFT_Slow_One_Dimension(x)
        else:
            fft_even = DFT.FFT_Cooley_Tukey(x[::2])
            fft_odd = DFT.FFT_Cooley_Tukey(x[1::2])
            fft_Cooley_Tukey = np.zeros(N, dtype=complex)
            half = int(N / 2)
            factor = np.exp(-2j * np.pi * np.arange(half) / N)
            for j in range(half):
                fft_Cooley_Tukey[j] = fft_even[j] + factor[j] * fft_odd[j]
                fft_Cooley_Tukey[j + half] = fft_even[j] - \
                    factor[j] * fft_odd[j]

            return fft_Cooley_Tukey

    @staticmethod
    def FFT_Cooley_Tukey_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]

        # Using Cooley-Tukey Inverse FFT algorithm.
        if N % 2 != 0:
            raise ValueError("Size of x must be a power of 2!")
        if N <= 2:
            return DFT.DFT_Slow_One_Dimension_Inverse(x)
        else:
            ifft_even = DFT.FFT_Cooley_Tukey_Inverse(x[::2])
            ifft_odd = DFT.FFT_Cooley_Tukey_Inverse(x[1::2])
            ifft_Cooley_Tukey = np.zeros(N, dtype=complex)
            half = int(N / 2)
            factor = np.exp(2j * np.pi * np.arange(half) / N)
            for j in range(half):
                ifft_Cooley_Tukey[j] = (
                    ifft_even[j] + factor[j] * ifft_odd[j]) * (half / N)
                ifft_Cooley_Tukey[j + half] = (ifft_even[j] -
                                               factor[j] * ifft_odd[j]) * (half / N)

            return ifft_Cooley_Tukey

    @staticmethod
    def DFT_Two_Dimensions(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        dft_two_dimensions = np.zeros((N, M), dtype=complex)

        # Using the 2D DFT algorithm.
        for i in range(N):
            for j in range(M):
                for k in range(M):
                    for l in range(N):
                        dft_two_dimensions[j, i] += x[k, l] * \
                            np.exp(-2j * np.pi * ((j * k / M) + (i * l / N)))

        return dft_two_dimensions

    @staticmethod
    def DFT_Two_Dimensions_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        idft_two_dimensions = np.zeros((N, M), dtype=complex)

        # Using the 2D Inverse DFT algorithm.
        for i in range(N):
            for j in range(M):
                for k in range(M):
                    for l in range(N):
                        idft_two_dimensions[j, i] += x[k, l] * np.exp(
                            2j * np.pi * ((j * k / M) + (i * l / N))) / (M * N)

        return idft_two_dimensions

    @staticmethod
    def FFT_Two_Dimensions(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        fft_two_dimensions = np.zeros((N, M), dtype=complex)

        # Applying the Cooley-Tuckey FFT to 2D.
        for n in range(N):
            fft_two_dimensions[n, :] = DFT.FFT_Cooley_Tukey(x[n, :])
        for m in range(M):
            fft_two_dimensions[:, m] = DFT.FFT_Cooley_Tukey(
                fft_two_dimensions[:, m])
        return fft_two_dimensions

    @staticmethod
    def FFT_Two_Dimensions_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        ifft_two_dimensions = np.zeros((N, M), dtype=complex)

        # Applying the Cooley-Tuckey Inverse FFT to 2D.
        for n in range(N):
            ifft_two_dimensions[n, :] = DFT.FFT_Cooley_Tukey_Inverse(x[n, :])
        for m in range(M):
            ifft_two_dimensions[:, m] = DFT.FFT_Cooley_Tukey_Inverse(
                ifft_two_dimensions[:, m])
        return ifft_two_dimensions
