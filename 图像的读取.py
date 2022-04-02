# This is a sample Python script.
import cv2 as cv
import numpy
import numpy as np
from random import randint


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


# 图像对象的创建
def mat_demo():
    image = cv.imread('D:/pictures/1.jpg')
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            b = randint(0, 255)
            g = randint(0, 255)
            r = randint(0, 255)
            image[row, col] = (~b, ~g, ~r)
    # roi = image[100:200, 100:200, :]
    # blank = np.zeros_like(image)
    # blank[100:200, 100:200, :] = image[100:200, 100:200, :]
    # cv.imshow("blank", blank)
    result = cv.imwrite('D:/pictures/2.jpg', image)
    print(result)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 滚动条的回调函数
def nothing(x):
    pass


# 滚动条调整图像亮度
def scroll_bar_demo():
    image = cv.imread('D:/pictures/1.jpg')
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", -100, 100, nothing)
    blank = np.zeros_like(image)
    cv.imshow("input", image)
    while True:
        pos = cv.getTrackbarPos("lightness", "input")
        blank[:, :] = [pos, pos, pos]
        result = cv.add(image, blank)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


# 滚动条调整图像亮度和对比度
def adjust_contrast_demo():
    image = cv.imread('D:/pictures/1.jpg')
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", 0, 100, nothing)
    cv.createTrackbar("contrast", "input", 0, 200, nothing)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        light = cv.getTrackbarPos("lightness", "input")
        contrast = cv.getTrackbarPos("contrast", "input") / 100
        print(f'light: {light}, contrast: {contrast}')
        blank[:, :] = [light, light, light]
        result = cv.addWeighted(image, contrast, blank, 0.5, light)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


# 按键触发事件
def keys_demo():
    image = cv.imread('D:/pictures/1.jpg')
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    while True:
        c = cv.waitKey(1)
        if c == 49:
            result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imshow("result", result)
        if c == 50:
            result = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("result", result)
        if c == 51:
            result = cv.bitwise_not(image)
            cv.imshow("result", result)
        if c == 27:
            break
    cv.destroyAllWindows()


# opencv自带颜色表操作
def color_table_demo():
    color_table = [cv.COLORMAP_AUTUMN,
                   cv.COLORMAP_BONE,
                   cv.COLORMAP_CIVIDIS,
                   cv.COLORMAP_COOL,
                   cv.COLORMAP_DEEPGREEN,
                   cv.COLORMAP_HOT,
                   cv.COLORMAP_HSV,
                   cv.COLORMAP_INFERNO,
                   cv.COLORMAP_JET,
                   cv.COLORMAP_MAGMA,
                   cv.COLORMAP_OCEAN,
                   cv.COLORMAP_PARULA,
                   cv.COLORMAP_PINK,
                   cv.COLORMAP_PLASMA,
                   cv.COLORMAP_RAINBOW,
                   cv.COLORMAP_SPRING,
                   cv.COLORMAP_SUMMER,
                   cv.COLORMAP_TURBO,
                   cv.COLORMAP_TWILIGHT,
                   cv.COLORMAP_TWILIGHT_SHIFTED,
                   cv.COLORMAP_VIRIDIS,
                   cv.COLORMAP_WINTER,
                   ]
    image = cv.imread('D:/pictures/1.jpg')
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    index = 0
    while True:
        dst = cv.applyColorMap(image, colormap=color_table[index % 19])
        index += 1
        cv.imshow("color_style", dst)
        c = cv.waitKey(500)
        if c == 27:
            break
    cv.destroyAllWindows()

# opencv逻辑操作
def logic_demo ():
    b1 = np.zeros((400, 400, 3), dtype=np.uint8)
    b1[:, :] = (255, 0, 255)
    b2 = np.zeros((400, 400, 3), dtype=np.uint8)
    b2[:, :] = (0, 255, 255)
    dst1 = cv.bitwise_and(b1, b2)
    dst2 = cv.bitwise_or(b1, b2)
    cv.imshow("b1", b1)
    cv.imshow("b2", b2)
    cv.imshow("dst1", dst1)
    cv.imshow("dst2", dst2)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 通道分离与合并 bgr 三个通道
def channels_demo():
    b1 = cv.imread('D:/pictures/1.jpg')
    cv.imshow("input", b1)
    # 将图片分割为三个通道，分别为 b mv[0] g mv[1] r mv[2]
    mv = cv.split(b1)
    mv[2][:, :] = 255
    result = cv.merge(mv)
    dst = np.zeros(b1.shape, dtype=np.uint8)
    cv.mixChannels([b1], [dst], fromTo=[2,0,1,1,0,2])
    print(b1)
    print(dst)
    cv.imshow("output4", dst)
    cv.waitKeyEx(0)
    cv.destroyAllWindows()

# 图像色彩空间转换
def color_space_convert_demo():
    b1 = cv.imread('D:/pictures/1.jpg')
    print(b1.shape)
    cv.imshow("input", b1)
    hsv = cv.cvtColor(b1, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    mask = cv.inRange(hsv, (11, 43, 46), (25, 255, 255))
    cv.imshow("mask", mask)
    cv.bitwise_not(mask, mask)
    result = cv.bitwise_and(b1, b1, mask=mask)
    cv.imshow("result", result)
    cv.waitKeyEx(0)
    cv.destroyAllWindows()

# 图像色彩空间转换
def pixel_stat_demo():
    b1 = cv.imread('D:/pictures/1.jpg')
    print(b1.shape)
    cv.imshow("input", b1)
    # means 为三个通道的均值 , dev 为方差
    means, dev = cv.meanStdDev(b1)
    print(f"means:\n{means}\ndev:\n{dev} ")
    cv.waitKeyEx(0)
    cv.destroyAllWindows()

# 基础图形的绘制
def drawing_demo():
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    b2 = cv.imread('D:/pictures/1.jpg')
    cv.rectangle(b1, (50, 50), (400, 400), (255, 0, 0), 2, 8, 0)
    cv.circle(b2, (400, 150), 100, (0, 0, 255), 2, 8, 0)
    cv.line(b1, (50, 50), (400, 400), (0, 255, 0), 2, 8, 0)
    cv.putText(b1, "99% face", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, 8)
    cv.imshow("rect", b1)
    cv.imshow("circle", b2)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 随机颜色
def random_color_demo():
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    while True:
        xx = np.random.randint(0, 512, 2, dtype=np.int32)
        yy = np.random.randint(0, 512, 2, dtype=np.int32)
        bgr = np.random.randint(0, 255, 3, dtype=np.int_)
        cv.line(b1, (xx[0], yy[0]), (xx[1], yy[1]), (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1, 8, 0)
        cv.imshow("line", b1)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()

# 多边形的填充与绘制
def polyline_drawing_demo():
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = numpy.array([[100, 100], [350, 100], [450, 280], [320, 450], [80, 400]], dtype=np.int32)
    # # 画多边形
    # cv.polylines(canvas, [pts], True, (0, 0, 255), 2, 8, 0)
    # # 填充的画多边形
    # cv.fillPoly(canvas, [pts], (0, 0, 255), 8, 0)
    cv.drawContours(canvas, [pts], -1, (255, 0, 0), 1)
    cv.imshow("polyline", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

b1 = np.zeros((512, 512, 3), dtype=np.uint8)

def mouse_drawing(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
    if event == cv.EVENT_MOUSEMOVE:
        pass
    if event == cv.EVENT_LBUTTONUP:
        print(x, y)

# 鼠标绘制图形
def mouse_drawing_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo", mouse_drawing)
    while True:
        cv.imshow("mouse_demo", b1)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mouse_drawing_demo()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
