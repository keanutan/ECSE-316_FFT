import numpy as np
import time


class DFT:

    @staticmethod
    def DFT_Slow_One_Dimension(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        dft = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                dft[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        return dft

    @staticmethod
    def DFT_Slow_One_Dimension_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]
        idft = np.zeros(N, dtype=complex)
        for n in range(N):
            for k in range(N):
                idft[n] += x[k] * np.exp(2j * np.pi * k * n / N) / N
        return idft

    @staticmethod
    def FFT_Cooley_Tukey(x):
        x = np.asarray(x, dtype=complex)
        N = x.shape[0]

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

    # @staticmethod
    # def FFT_Cooley_Tukey(x):
    #     x = np.asarray(x, dtype=complex)
    #     N = x.shape[0]

    #     if N == 1:
    #         return [x[0]]

    #     fft_even = DFT.FFT_Cooley_Tukey(x[::2])
    #     fft_odd = DFT.FFT_Cooley_Tukey(x[1::2])
    #     fft_Cooley_Tukey = np.zeros(N, dtype=complex)
    #     half = int(N / 2)
    #     # factor = np.exp(2j * np.pi * np.arange(half) / N)
    #     factor = np.exp(-2j * np.pi * np.arange(half) / N)
    #     for j in range(half):
    #         fft_Cooley_Tukey[j] = fft_even[j] + factor[j] * fft_odd[j]
    #         fft_Cooley_Tukey[j + half] = fft_even[j] - \
    #             factor[j] * fft_odd[j]
    #     return fft_Cooley_Tukey

    # @staticmethod
    # def ifft(X):
    #     X = np.asarray(X, dtype=complex)
    #     N = X.shape[0]

    #     if N == 1:
    #         return [X[0]]

    #     fft_Cooley_Tukey_inverse = np.zeros(N, dtype=complex)

    #     even = DFT.ifft(X[:N:2])
    #     odd = DFT.ifft(X[1:N:2])

    #     for k in range(N//2):
    #         w = np.exp(2j * np.pi * k / N)
    #         fft_Cooley_Tukey_inverse[k] = (even[k] + w * odd[k])/N
    #         fft_Cooley_Tukey_inverse[k + N//2] = (even[k] - w * odd[k])/N
    #     return fft_Cooley_Tukey_inverse

    # @staticmethod
    # def FFT_Cooley_Tukey_Inverse(x):
    #     x = np.asarray(x, dtype=complex)
    #     N = x.shape[0]

    #     if N == 1:
    #         return [x[0]]

    #     fft_even = DFT.FFT_Cooley_Tukey_Inverse(x[::2])
    #     fft_odd = DFT.FFT_Cooley_Tukey_Inverse(x[1::2])
    #     fft_Cooley_Tukey = np.zeros(N, dtype=complex)
    #     half = int(N / 2)
    #     for j in range(half):
    #         factor = np.exp(2j * np.pi * j / N)
    #         fft_Cooley_Tukey[j] = (fft_even[j] + factor * fft_odd[j]) / N
    #         fft_Cooley_Tukey[j + half] = (fft_even[j] - factor * fft_odd[j]) / N

    #     return fft_Cooley_Tukey

    # @staticmethod
    # def FFT_Cooley_Tukey_Inverse(x):
    #     x = np.asarray(x, dtype=complex)
    #     N = x.shape[0]
    #     if N % 2 != 0:
    #         raise ValueError("Size of x must be a power of 2.")
    #     elif N <= 8:
    #         return DFT_Slow_One_Dimension_Inverse(x)
    #     else:
    #         fft_even = FFT_Cooley_Tukey_Inverse(x[::2])
    #         fft_odd = FFT_Cooley_Tukey_Inverse(x[1::2])
    #         fft_Cooley_Tukey = np.empty(N, dtype=complex)
    #         half = int(N/2)
    #         for n in range(N):
    #             fft_Cooley_Tukey[n] = (
    #                 fft_even[n % half] + np.exp(-2j * np.pi * n / N) * fft_odd[n % half]) * (half / N)
    #     return fft_Cooley_Tukey

    @staticmethod
    def DFT_Two_Dimensions(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        dft_two_dimensions = np.zeros((N, M), dtype=complex)

        for i in range(N):
            for j in range(M):
                for k in range(M):
                    for l in range(N):
                        dft_two_dimensions[j, i] += x[k, l] * np.exp(-2j * np.pi *((j * k / M) + (i * l / N)))

        return dft_two_dimensions

    @staticmethod
    def DFT_Two_Dimensions_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        idft_two_dimensions = np.zeros((N, M), dtype=complex)

        for i in range(N):
            for j in range(M):
                for k in range(M):
                    for l in range(N):
                        idft_two_dimensions[j, i] += x[k, l] * np.exp(2j * np.pi *((j * k / M) + (i * l / N))) / (M * N)

        return idft_two_dimensions

    @staticmethod
    def FFT_Two_Dimensions(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        fft_two_dimensions = np.zeros((N, M), dtype=complex)

        for n in range(N):
            fft_two_dimensions[n, :] = DFT.FFT_Cooley_Tukey(x[n, :])
        for m in range(M):
            fft_two_dimensions[:, m] = DFT.FFT_Cooley_Tukey(fft_two_dimensions[:, m])
        return fft_two_dimensions

    @staticmethod
    def FFT_Two_Dimensions_Inverse(x):
        x = np.asarray(x, dtype=complex)
        N, M = x.shape
        ifft_two_dimensions = np.zeros((N, M), dtype=complex)

        for n in range(N):
            ifft_two_dimensions[n, :] = DFT.FFT_Cooley_Tukey_Inverse(x[n, :])
        for m in range(M):
            ifft_two_dimensions[:, m] = DFT.FFT_Cooley_Tukey_Inverse(ifft_two_dimensions[:, m])
        return ifft_two_dimensions

    @staticmethod
    def test():
        print(str(int(np.random.random()*1000)))
        # x = np.random.random(5)
        # a = np.zeros(8)
        # for item in range(8):
        #     try:
        #         a[item] = x[item]
        #     except IndexError as e:
        #         a[item] = 0
        #         pass
            
        # print(np.allclose(DFT_Slow_One_Dimension(x), np.fft.fft(x)))
        # print(np.allclose(fast_one_dimension(x), np.fft.fft(x)))
        # print(np.allclose(DFT_Slow_One_Dimension_Inverse(x), np.fft.ifft(x)))

        # start = time.time()
        # print("np fft")
        # print(np.fft.fft(a))
        # # np.fft.fft(x)
        # end = time.time()
        # print(end - start)

        # print("np ifft")
        # start = time.time()
        # print(np.fft.ifft(a))
        # # np.fft.ifft(x)
        # end = time.time()
        # print(end - start)

        # print("hello")
        # start = time.time()
        # print("DFT_Slow_One_Dimension")
        # print(DFT.DFT_Slow_One_Dimension(x))
        # DFT.DFT_Slow_One_Dimension(x)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("DFT_Slow_One_Dimension_Inverse")
        # print(DFT.DFT_Slow_One_Dimension_Inverse(x))
        # DFT.DFT_Slow_One_Dimension_Inverse(x)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey: 2")
        # # print(DFT.FFT_Cooley_Tukey(x, 2))
        # print(DFT.FFT_Cooley_Tukey(a))
        # # DFT.FFT_Cooley_Tukey(x,2)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey : 4")
        # print(DFT.FFT_Cooley_Tukey(x, 4))
        # # DFT.FFT_Cooley_Tukey(x,2)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey: 8")
        # print(DFT.FFT_Cooley_Tukey(x, 8))
        # # DFT.FFT_Cooley_Tukey(x,8)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey: 16")
        # print(DFT.FFT_Cooley_Tukey(x, 16))
        # # DFT.FFT_Cooley_Tukey(x,16)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey: 32")
        # print(DFT.FFT_Cooley_Tukey(x, 32))
        # # DFT.FFT_Cooley_Tukey(x,32)
        # end = time.time()
        # print(end - start)

        # print("FFT_Cooley_Tukey_Inverse: 2")
        # start = time.time()
        # print(DFT.FFT_Cooley_Tukey_Inverse(a))
        # end = time.time()
        # print(end - start)

        # print("FFT_Cooley_Tukey_Inverse: 4")
        # start = time.time()
        # print(DFT.FFT_Cooley_Tukey_Inverse(x, 4))
        # end = time.time()
        # print(end - start)

        # print("FFT_Cooley_Tukey_Inverse: 8")
        # start = time.time()
        # print(DFT.FFT_Cooley_Tukey_Inverse(x, 8))
        # end = time.time()
        # print(end - start)

        # print("FFT_Cooley_Tukey_Inverse: 16")
        # start = time.time()
        # print(DFT.FFT_Cooley_Tukey_Inverse(x, 16))
        # end = time.time()
        # print(end - start)

        # print("FFT_Cooley_Tukey_Inverse: 32")
        # start = time.time()
        # print(DFT.FFT_Cooley_Tukey_Inverse(x, 32))
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # print("FFT_Cooley_Tukey_Inverse___")
        # print(DFT.ifft(x))
        # end = time.time()
        # print(end - start)

        # print(fast_one_dimension(x))

        a2 = np.random.rand(4, 4)
        # anew = np.zeros((4,8))
        # anew[:4, :5] = a2
        # fft2 = np.fft.fft2(a2)
        # fft2d = DFT.FFT_Two_Dimensions(a2)
        # fft2dd = DFT.DFT_Two_Dimensions(a2)
        ifft2 = np.fft.ifft2(a2)
        ifft2d = DFT.FFT_Two_Dimensions_Inverse(a2)
        ifft2d = DFT.DFT_Two_Dimensions_Inverse(a2)

        # print("not mine")
        # print(fft2)
        # print("mine")
        # print(fft2d)
        # print(fft2dd)

        print("inverse not mine")
        print(ifft2)
        print("inverse mine")
        print(ifft2d)


def main():
    print("Hello World!")
    DFT.test()


if __name__ == "__main__":
    main()
