import numpy as np
import timeit


# class DFT:

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

def FFT_Cooley_Tukey(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2.")
    elif N <= 8:
        return DFT_Slow_One_Dimension(x)
    else:
        fft_even = FFT_Cooley_Tukey(x[::2])
        fft_odd = FFT_Cooley_Tukey(x[1::2])
        fft_Cooley_Tukey = np.empty(N, dtype=complex)
        half = int(N/2)
        for k in range(N):
            fft_Cooley_Tukey[k] = fft_even[k % half] + np.exp(-2j * np.pi * k / N) * fft_odd[k % half]
    return fft_Cooley_Tukey

def FFT_Cooley_Tukey_Inverse(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError("Size of x must be a power of 2.")
    elif N <= 8:
        return DFT_Slow_One_Dimension_Inverse(x)
    else:
        fft_even = FFT_Cooley_Tukey_Inverse(x[::2])
        fft_odd = FFT_Cooley_Tukey_Inverse(x[1::2])
        fft_Cooley_Tukey = np.empty(N, dtype=complex)
        half = int(N/2)
        for n in range(N):
            fft_Cooley_Tukey[n] = (fft_even[n % half] + np.exp(-2j * np.pi * n / N) * fft_odd[n % half]) * (half / N)
    return fft_Cooley_Tukey



            # for n in range(int(N/2)):
            #     factor = np.exp(-2j * np.pi * n / N) * fft_odd[n]
            #     fft_Cooley_Tukey[n] = fft_even[n] + factor
            #     fft_Cooley_Tukey[N/2 + n] = fft_even[n] - factor
            # return fft_Cooley_Tukey

    # def fast_one_dimension(a):
    #     a = np.asarray(a, dtype=complex)
    #     N = a.shape[0]
    #     if N % 2 != 0:
    #         raise AssertionError("size of a must be a power of 2")
    #     elif N <= 16:
    #         return DFT_Slow_One_Dimension(a)
    #     else:
    #         even = fast_one_dimension(a[::2])
    #         odd = fast_one_dimension(a[1::2])
    #         res = np.zeros(N, dtype=complex)
    #         half_size = N//2
    #         for n in range(N):
    #             res[n] = even[n % half_size] + \
    #                 np.exp(-2j * np.pi * n / N) * odd[n % half_size]
    #     return res

        # l = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        # for x in l:
        #     print(l[::2], l[1::2])

x = np.random.random(1024)
# print(np.allclose(DFT_Slow_One_Dimension(x), np.fft.fft(x)))
# print(np.allclose(fast_one_dimension(x), np.fft.fft(x)))
# print(np.allclose(DFT_Slow_One_Dimension_Inverse(x), np.fft.ifft(x)))
print(np.fft.ifft(x))
# print("hello")
print(DFT_Slow_One_Dimension(x))
print(DFT_Slow_One_Dimension_Inverse(x))
# print(fast_one_dimension(x))
print(FFT_Cooley_Tukey(x))
print(FFT_Cooley_Tukey_Inverse(x))