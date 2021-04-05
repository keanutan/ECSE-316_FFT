import argparse
import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np


from scipy.sparse import csr_matrix, save_npz
from matplotlib import colors

from dft import DFT


def closestPowerOfTwo(n):
    return 2**(n-1).bit_length()


def main():

    # im_raw = plt.imread("moonlanding.png").astype(float)
    # print(im_raw.shape)

    command = None
    try:
        command = parser()
    except BaseException as e:
        print("ERROR\t Incorrect command syntax")
        return

    mode = command.mode
    image = command.image

    if mode == 1:

        # Opening the image.
        init = Image.open(image)
        init = ImageOps.grayscale(init)
        img = np.array(init)

        # Computing the dimensions of the image array.
        initial_shape = img.shape

        # Creating a new array with dimensions that are powers of 2.
        # Populating that new array with the img array.
        updated_image = np.zeros(
            (closestPowerOfTwo(img.shape[0]), closestPowerOfTwo(img.shape[1])))
        updated_image[:img.shape[0], :img.shape[1]] = img

        # Computing the 2D FFT of the image.
        transient_img = DFT.FFT_Two_Dimensions(updated_image)

        # Converting the array into an image for resizing using PIL.
        temp = Image.fromarray(np.log(1+np.abs(transient_img)))

        # Resizing image to original size and converting back into array to be able to .imshow()
        final_img = temp.resize((img.shape[1], img.shape[0]), Image.ANTIALIAS)
        final_img = np.array(final_img)

        # Creating the plot presentation.
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img, "gray")
        ax[0].set_title("Original")
        ax[1].imshow(final_img, "gray")
        ax[1].set_title("2D FFT")
        fig.suptitle("Mode 1: 2-Dimensional Fast Fourier Transform of Image")
        plt.show()

    elif mode == 2:

        # Opening the image.
        init = Image.open(image)
        init = ImageOps.grayscale(init)
        img = np.array(init)

        # Computing the dimensions of the image array.
        initial_shape = img.shape

        # Creating a new array with dimensions that are powers of 2.
        # Populating that new array with the img array.
        updated_image = np.zeros(
            (closestPowerOfTwo(img.shape[0]), closestPowerOfTwo(img.shape[1])))
        updated_image[:img.shape[0], :img.shape[1]] = img

        # Computing the 2D FFT of the image.
        transient_img = DFT.FFT_Two_Dimensions(updated_image)
        I, J = transient_img.shape
        threshold = 0.09
        transient_img[int(I*threshold):int(I*(1-threshold))] = 0
        transient_img[:, int(J*threshold):int(J*(1-threshold))] = 0

        # Computing the Inverse 2D FFT of the image.
        transient_img = DFT.FFT_Two_Dimensions_Inverse(transient_img).real
        denoised_img = transient_img[:img.shape[0], :img.shape[1]]

        # Printing denoising data.
        threshold_inverse = 1-threshold
        print("Fraction of Non-Zero Coefficients {} Representing ({}, {}) out of ({}, {}) Pixels".format(threshold_inverse**2, int(I*threshold_inverse), int(J*threshold_inverse), I, J))

        # Creating the plot presentation.
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img, "gray")
        ax[0].set_title("Original")
        ax[1].imshow(denoised_img, "gray")
        ax[1].set_title("Denoised Image")
        fig.suptitle(
            "Mode 2: Denoised Image Using 2-Dimensional Fast Fourier Transform")
        plt.show()

    elif mode == 3:

        # Opening the image.
        init = Image.open(image)
        init = ImageOps.grayscale(init)
        img = np.array(init)

        # Computing the dimensions of the image array.
        initial_shape = img.shape

        # Creating a new array with dimensions that are powers of 2.
        # Populating that new array with the img array.
        updated_image = np.zeros(
            (closestPowerOfTwo(img.shape[0]), closestPowerOfTwo(img.shape[1])))
        updated_image[:img.shape[0], :img.shape[1]] = img

        # Computing the 2D FFT of the image.
        transient_img_a = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_b = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_c = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_d = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_e = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_f = DFT.FFT_Two_Dimensions(updated_image)
        abs_values = np.sort(np.abs(transient_img_a.flatten()))
        percentile_a = 0.20 * abs_values.shape[0]
        percentile_b = 0.40 * abs_values.shape[0]
        percentile_c = 0.80 * abs_values.shape[0]
        percentile_d = 0.80 * abs_values.shape[0]
        percentile_e = 0.95 * abs_values.shape[0]
        percentile_f = abs_values.shape[0] - 1
        cutoff_a = abs_values[int(percentile_a)]
        cutoff_b = abs_values[int(percentile_b)]
        cutoff_c = abs_values[int(percentile_c)]
        cutoff_d = abs_values[int(percentile_d)]
        cutoff_e = abs_values[int(percentile_e)]
        cutoff_f = abs_values[int(percentile_f)]
        # print(cutoff)

        # cutoff = np.percentile(transient_img, 15)


        np.place(transient_img_a, abs(transient_img_a) < cutoff_a, [0])
        np.place(transient_img_b, abs(transient_img_b) < cutoff_b, [0])
        np.place(transient_img_c, abs(transient_img_c) < cutoff_c, [0])
        np.place(transient_img_d, abs(transient_img_d) < cutoff_d, [0])
        np.place(transient_img_e, abs(transient_img_e) < cutoff_e, [0])
        np.place(transient_img_f, abs(transient_img_f) < cutoff_f, [0])

        I, J = transient_img_a.shape
        # for i in range(I):
        #     for j in range(J):
        #         if np.abs(transient_img_a[i,j]) > cutoff_a:
        #             transient_img_a[i, j] = [0]
        #         if np.abs(transient_img_b[i,j]) > cutoff_b:
        #             transient_img_b[i, j] = [0]
        #         if np.abs(transient_img_c[i,j]) > cutoff_c:
        #             transient_img_c[i, j] = [0]
        #         if np.abs(transient_img_d[i,j]) > cutoff_d:
        #             transient_img_d[i, j] = [0]
        #         if np.abs(transient_img_e[i,j]) > cutoff_e:
        #             transient_img_e[i, j] = [0]
        #         if np.abs(transient_img_f[i,j]) > cutoff_f:
        #             transient_img_f[i, j] = [0]

        count_a = 0
        count_b = 0
        count_c = 0
        count_d = 0
        count_e = 0
        count_f = 0
        
        for i in range(I):
            for j in range(J):
                if np.abs(transient_img_a[i,j]) == 0:
                    count_a+=1
                if np.abs(transient_img_b[i,j]) == 0:
                    count_b+=1
                if np.abs(transient_img_c[i,j]) == 0:
                    count_c+=1
                if np.abs(transient_img_d[i,j]) == 0:
                    count_d+=1
                if np.abs(transient_img_e[i,j]) == 0:
                    count_e+=1
                if np.abs(transient_img_f[i,j]) == 0:
                    count_f+=1

        # print("count a: {} ".format(count_a))
        # print("count a: {} ".format(count_b))
        # print("count a: {} ".format(count_c))
        # print("count a: {} ".format(count_d))
        # print("count a: {} ".format(count_e))
        # print("count a: {} ".format(count_f))
                
        print("For 20% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_a))
        print("For 40% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_b))
        print("For 60% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_c))
        print("For 80% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_d))
        print("For 95% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_e))
        # print("For 20% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_f))
            




        # threshold = 0.09
        # transient_img[int(I*threshold):int(I*(1-threshold))] = 0
        # transient_img[:, int(J*threshold):int(J*(1-threshold))] = 0
        name = "compressed_" + str(int(20)) + ".txt"
        save_npz(name, csr_matrix(transient_img_a))
        # np.savetxt(name, transient_img_a, delimiter=",")
        name = "compressed_" + str(int(40)) + ".txt"
        save_npz(name, csr_matrix(transient_img_b))
        # np.savetxt(name, transient_img_b, delimiter=",")
        name = "compressed_" + str(int(60)) + ".txt"
        save_npz(name, csr_matrix(transient_img_c))
        # np.savetxt(name, transient_img_c, delimiter=",")
        name = "compressed_" + str(int(80)) + ".txt"
        save_npz(name, csr_matrix(transient_img_d))
        # np.savetxt(name, transient_img_d, delimiter=",")
        name = "compressed_" + str(int(95)) + ".txt"
        save_npz(name, csr_matrix(transient_img_e))
        # np.savetxt(name, transient_img_e, delimiter=",")

        # Computing the Inverse 2D FFT of the image.
        done_a = DFT.FFT_Two_Dimensions_Inverse(transient_img_a).real
        compressed_img_a = done_a[:img.shape[0], :img.shape[1]]

        done_b = DFT.FFT_Two_Dimensions_Inverse(transient_img_b).real
        compressed_img_b = done_b[:img.shape[0], :img.shape[1]]

        done_c = DFT.FFT_Two_Dimensions_Inverse(transient_img_c).real
        compressed_img_c = done_c[:img.shape[0], :img.shape[1]]

        done_d = DFT.FFT_Two_Dimensions_Inverse(transient_img_d).real
        compressed_img_d = done_d[:img.shape[0], :img.shape[1]]

        done_e = DFT.FFT_Two_Dimensions_Inverse(transient_img_e).real
        compressed_img_e = done_e[:img.shape[0], :img.shape[1]]

        # name = "foo" + str(fileNumber) + ".csv"
        # done_f = DFT.FFT_Two_Dimensions_Inverse(transient_img_f).real
        # compressed_img_f = done_f[:img.shape[0], :img.shape[1]]

        # # Printing denoising data.
        # threshold_inverse = 1-threshold
        # print("Fraction of Non-Zero Coefficients {} Representing ({}, {}) out of ({}, {}) Pixels".format(threshold_inverse**2, int(I*threshold_inverse), int(J*threshold_inverse), I, J))

        # Creating the plot presentation.
        fig, ax = plt.subplots(2, 3)
        ax[0,0].imshow(img, "gray")
        ax[0,0].set_title("Original")

        ax[0,1].imshow(compressed_img_a, "gray")
        ax[0,1].set_title("20% Compression")

        ax[0,2].imshow(compressed_img_b, "gray")
        ax[0,2].set_title("40% Compression")

        ax[1,0].imshow(compressed_img_c, "gray")
        ax[1,0].set_title("60% Compression")

        ax[1,1].imshow(compressed_img_d, "gray")
        ax[1,1].set_title("80% Compression")

        ax[1,2].imshow(compressed_img_e, "gray")
        ax[1,2].set_title("95% Compression")

        # ax[1,3].imshow(compressed_img_f, "gray")
        # ax[1,3].set_title("100% Compression")

        fig.suptitle(
            "Mode 3: Compression")
        plt.show()

    elif mode == 4:
        print("hello")
    else:
        print("hello")


def parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', action='store', dest='mode', help='Mode of operation 1-> fast, 2-> denoise, 3-> compress&save 4-> plot', type=int, default=1)
    parser.add_argument('-m', action='store', dest='mode',
                        help='- [1] (Default) for fast mode where the image is converted into its FFT form and displayed \n- [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed \n- [3] for compressing and saving the image \n- [4] for plotting the runtime graphs for the report', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='Filename of the image we wish to take the DFT of', type=str, default='moonlanding.png')
    # parser.add_argument('-i', action='store', dest='image', help='image path to work on', type=str, default='moonlanding.png')
    return parser.parse_args()

# def parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', action='store', dest='mode',
#                         help='Mode of operation 1-> fast, 2-> denoise, 3-> compress&save 4-> plot', type=int, default=1)
#     parser.add_argument('-i', action='store', dest='image',
#                         help='image path to work on', type=str, default='moonlanding.png')
#     return parser.parse_args()


if __name__ == "__main__":
    main()
