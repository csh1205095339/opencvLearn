# This is a sample Python script.
import cv2 as cv
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 图像的读取与显示
def readImgDemo():
    image = cv.imread('D:/pictures/1.jpg')
    cv.imshow("input", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 图像色彩空间
def color_space_demo():
    image = cv.imread('D:/pictures/1.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("gray", gray)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()

def mat_demo():
    image = cv.imread('D:/pictures/1.jpg')
    print(image.shape)
    roi = image[100:200, 100:200, :]
    blank = np.zeros_like(image)
    blank[100:200, 100:200, :] = image[100:200, 100:200, :]
    cv.imshow("blank", blank)
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # Press Ctrl+F8 to toggle the breakpoint.
    print(f'Hi, {name}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mat_demo()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
