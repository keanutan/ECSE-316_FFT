import argparse
import math
import time

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

    # Getting command line input.
    command = None
    try:
        command = parser()
    except BaseException as e:
        print("ERROR\t Incorrect command syntax")
        return

    mode = command.mode
    image = command.image

    # Mode 1
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

    # Mode 2
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
        print("Fraction of Non-Zero Coefficients {} Representing ({}, {}) out of ({}, {}) Pixels".format(
            threshold_inverse**2, int(I*threshold_inverse), int(J*threshold_inverse), I, J))

        # Creating the plot presentation.
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img, "gray")
        ax[0].set_title("Original")
        ax[1].imshow(denoised_img, "gray")
        ax[1].set_title("Denoising Using Only Cutoff Coefficient of 0.09")
        fig.suptitle(
            "Mode 2: Denoised Image Using 2-Dimensional Fast Fourier Transform")
        plt.show()

    # Mode 3
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

        # Creating multiple 2D arrays for all compression levels.
        transient_img_a = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_b = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_c = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_d = DFT.FFT_Two_Dimensions(updated_image)
        transient_img_e = DFT.FFT_Two_Dimensions(updated_image)

        # Computing the threshold percentile value that will be used to set coefficients equal to 0.
        abs_values = np.sort(np.abs(transient_img_a.flatten()))
        percentile_a = 0.20 * abs_values.shape[0]
        percentile_b = 0.40 * abs_values.shape[0]
        percentile_c = 0.60 * abs_values.shape[0]
        percentile_d = 0.80 * abs_values.shape[0]
        percentile_e = 0.95 * abs_values.shape[0]
        percentile_f = abs_values.shape[0] - 1
        cutoff_a = abs_values[int(percentile_a)]
        cutoff_b = abs_values[int(percentile_b)]
        cutoff_c = abs_values[int(percentile_c)]
        cutoff_d = abs_values[int(percentile_d)]
        cutoff_e = abs_values[int(percentile_e)]

        # Iterating through each of the 2D arrays and setting coefficients that are under the cutoff equal to zero.
        np.place(transient_img_a, abs(transient_img_a) < cutoff_a, [0])
        np.place(transient_img_b, abs(transient_img_b) < cutoff_b, [0])
        np.place(transient_img_c, abs(transient_img_c) < cutoff_c, [0])
        np.place(transient_img_d, abs(transient_img_d) < cutoff_d, [0])
        np.place(transient_img_e, abs(transient_img_e) < cutoff_e, [0])

        # Counting the number of zeros in the 2D arrays.
        count_a = 0
        count_b = 0
        count_c = 0
        count_d = 0
        count_e = 0
        count_f = 0

        I, J = transient_img_a.shape
        for i in range(I):
            for j in range(J):
                if np.abs(transient_img_a[i, j]) == 0:
                    count_a += 1
                if np.abs(transient_img_b[i, j]) == 0:
                    count_b += 1
                if np.abs(transient_img_c[i, j]) == 0:
                    count_c += 1
                if np.abs(transient_img_d[i, j]) == 0:
                    count_d += 1
                if np.abs(transient_img_e[i, j]) == 0:
                    count_e += 1

        # Printing the number of non-zero Fourier coefficients for each compression level.
        print("For 20% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_a))
        print("For 40% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_b))
        print("For 60% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_c))
        print("For 80% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_d))
        print("For 95% compression, we have {} non-zero Fourier coefficients.".format(524288 - count_e))

        # Saving compressed arrays as .txt files and .npz files to better see file size difference.
        name = "compressed_" + str(int(20)) + ".txt"
        np.savetxt(name, transient_img_a)
        name = "compressed_" + str(int(20))
        save_npz(name, csr_matrix(transient_img_a))

        name = "compressed_" + str(int(40)) + ".txt"
        np.savetxt(name, transient_img_b)
        name = "compressed_" + str(int(40))
        save_npz(name, csr_matrix(transient_img_b))

        name = "compressed_" + str(int(60)) + ".txt"
        np.savetxt(name, transient_img_c)
        name = "compressed_" + str(int(60))
        save_npz(name, csr_matrix(transient_img_c))

        name = "compressed_" + str(int(80)) + ".txt"
        np.savetxt(name, transient_img_d)
        name = "compressed_" + str(int(80))
        save_npz(name, csr_matrix(transient_img_d))

        name = "compressed_" + str(int(95)) + ".txt"
        np.savetxt(name, transient_img_e)
        name = "compressed_" + str(int(95))
        save_npz(name, csr_matrix(transient_img_e))

        # Computing the Inverse 2D FFT of the compressed image.
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

        # Creating the plot presentation.
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(img, "gray")
        ax[0, 0].set_title("Original")

        ax[0, 1].imshow(compressed_img_a, "gray")
        ax[0, 1].set_title("20% Compression")

        ax[0, 2].imshow(compressed_img_b, "gray")
        ax[0, 2].set_title("40% Compression")

        ax[1, 0].imshow(compressed_img_c, "gray")
        ax[1, 0].set_title("60% Compression")

        ax[1, 1].imshow(compressed_img_d, "gray")
        ax[1, 1].set_title("80% Compression")

        ax[1, 2].imshow(compressed_img_e, "gray")
        ax[1, 2].set_title("95% Compression")

        fig.suptitle(
            "Mode 3: Compression")
        plt.show()

    # Mode 4
    elif mode == 4:

        # Setting up the plot.
        fig, ax = plt.subplots()
        ax.set_xlabel('Problem Size NxN')
        ax.set_ylabel('Runtime In Seconds')
        ax.set_title('Mode 4: Runtime Over Problem Size')

        # Setting up our testing parameters with initial exponent, the number of trials per input,
        # the counter for adding plot points at each exponent, and the initial data arrays for
        # 2D FFT and 2D DFT.

        exponent = 5
        trials = 10
        counter = 0
        x_axis_data = np.zeros(6)
        fft_data = np.zeros(6)
        fft_data_std = np.zeros(6)

        # Note: We only take two plot points for the 2D DFT because computing times for inputs with size >= 2^7
        #       take extremely long to compute.
        dft_data = np.zeros(2)
        dft_data_std = np.zeros(2)

        # While loop that goes through inputs of size 2^5 to 2^10.
        while exponent <= 10:

            # Anouncing start of trials for particular input.
            print("Starting trials with input of size " +
                  str(2**exponent) + "x" + str(2**exponent))

            # Data arrays for both 2D Fourier Transforms.
            runtimes_dft = np.zeros(10)
            runtimes_fft = np.zeros(10)
            
            # Setting up the random 2D array of size 2^exponent x 2^exponent.
            x = np.random.rand(2**exponent, 2**exponent)

            # Doing 10 independent trials with 10 different inputs.
            for i in range(10):
                
                # Computing runtime for 2D FFT.
                start = time.time()
                DFT.FFT_Two_Dimensions(x)
                end = time.time()
                runtimes_fft[i] = (end - start)

                # If statement to prevent computation of 2D DFT as it takes too long for inputs of size 128x128.
                if exponent <= 6:
                    # Computing runtime for 2D DFT.
                    start = time.time()
                    DFT.DFT_Two_Dimensions(x)
                    end = time.time()
                    runtimes_dft[i] = (end - start)

                print("\tRunning trial " + str(i))

            # Computing average and standart deviation of runtimes.
            average_runtime_fft = np.average(runtimes_fft)
            std_runtime_fft = np.std(runtimes_fft)

            # Storing the data points in the fft_data and fft_data_std arrays.
            fft_data[counter] = average_runtime_fft
            fft_data_std[counter] = std_runtime_fft

            # If statement to prevent computation of 2D DFT as it takes too long for inputs of size 128x128.
            if exponent <= 6:
                # Computing average and standart deviation of runtimes.
                average_runtime_dft = np.average(runtimes_dft)
                std_runtime_dft = np.std(runtimes_dft)

                # Storing the data points in the dft_data and dft_data_std arrays.
                dft_data[counter] = average_runtime_dft
                dft_data_std[counter] = std_runtime_dft

                # Printing Average Runtime and Standard Deviation for 2D DFT.
                print("\t2D DFT:\tAverage Runtime: " + str(average_runtime_dft) +
                      "\tRuntime Standard Deviation: " + str(std_runtime_dft))

            # Storing input sizes for the x-axis.
            x_axis_data[counter] = 2**exponent

            # Printing Average Runtime and Standard Deviation for 2D FFT.
            print("\t2D FFT:\tAverage Runtime: " + str(average_runtime_fft) +
                  "\tRuntime Standard Deviation: " + str(std_runtime_fft) + "\n")

            # Incrementing counter and exponent.
            counter += 1
            exponent += 1

        # Plotting both 2D FFT and 2D DFT.
        plt.errorbar(x_axis_data, fft_data, yerr=(fft_data_std*2), fmt='r--')
        plt.errorbar(x_axis_data[0:2], dft_data,
                     yerr=(dft_data_std*2), fmt='g--')
        plt.show()

    # Invalid Mode.
    else:
        print("Error:\t There are only 4 modes that this program works in.")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='mode',
                        help='- [1] (Default) for fast mode where the image is converted into its FFT form and displayed \n- [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed \n- [3] for compressing and saving the image \n- [4] for plotting the runtime graphs for the report', type=int, default=1)
    parser.add_argument('-i', action='store', dest='image',
                        help='Filename of the image we wish to take the DFT of', type=str, default='moonlanding.png')
    return parser.parse_args()


if __name__ == "__main__":
    main()
