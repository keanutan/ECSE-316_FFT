import numpy as np
import timeit

class DFT:

    def DFT_Slow_One_Dimension(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        dft = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                dft[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

        return dft

    def DFT_Slow_One_Dimension_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        idft = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                idft[n] += x[k] * np.exp(-2j * np.pi * k * n / N)

            idft[n] /= N

        return idft


    x = np.random.random(1024)


    print(np.allclose(DFT_Slow_One_Dimension(x), np.fft.fft(x)))
    print(np.allclose(DFT_Slow_One_Dimension_Inverse(x), np.fft.ifft(x)))

    print(np.fft.ifft(x))
    print(DFT_Slow_One_Dimension_Inverse(x))

