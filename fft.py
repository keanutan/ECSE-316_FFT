import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from dft import DFT

def gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def main():
    # command = None
    # try:
    #     command = parser()
    # except BaseException as e:
    #     print("ERROR:\t Incorrect Input")
    #     return


    # # print("Hello World!")
    # # DFT.test()

    # mode = command.mode
    # image = command.image

    # if mode == 1:
    img = mpimg.imread('moonlanding.png')
    # new_img = gray(img)
    # new_img = np.asa
    # new_img = img[:,:,0]

    # old_shape = img.shape
    # new_shape = des
    # indices = np.dstack(np.indices(img.shape[:2]))
    # data = np.concatenate((img, indices), axis=-1)

    # im_fft = DFT.FFT_Two_Dimensions(data)
    plt.imshow(img)
    plt.show()
    # img = mpimg.imread('moonlanding.png')
    # plt.imshow(img)



# def parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', action='store', dest='mode',
#                         help='- [1] (Default) for fast mode where the image is converted into its FFT form and displayed - [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed - [3] for compressing and saving the image - [4] for plotting the runtime graphs for the report', type=int, default=1)
#     parser.add_argument('-i', action='store', dest='image',
#                         help='Filename of the image we wish to take the DFT of', type='str', default='moonlanding.png')
#     return parser.parse_args()


if __name__ == "__main__":
    main()
