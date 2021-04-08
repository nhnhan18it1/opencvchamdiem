import cv2 as cv
import imutils
from math import ceil
import matplotlib.pyplot as plt
import numpy

def get_x_v1(s):
    s = cv.boundingRect(s)
    return s[0] * s[1]


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def crop_img(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred = cv.GaussianBlur(gray_img, (5, 5), 0)

    img_canny = cv.Canny(blurred, 100, 200)

    cnts = cv.findContours(img_canny.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print(cnts)
    # cv.imshow("graay", img_canny)
    # cv.waitKey(0)
    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    if len(cnts)>0:
        cnts = sorted(cnts, key=get_x_v1)
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv.boundingRect(c)
            if w_curr * h_curr > 100000:
                check_xy_min = x_curr*y_curr-x_old*y_old
                check_xy_max = (x_curr+w_curr)*(y_curr+h_curr)-(x_old+w_old)*(y_old+h_old)

                if len(ans_blocks) == 0:
                    ans_blocks.append((gray_img[y_curr:y_curr+h_curr, x_curr:x_curr+w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append((gray_img[y_curr:y_curr+h_curr, x_curr:x_curr+w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
    sorted_ans_blocks = sorted(ans_blocks, key=get_x)
    return sorted_ans_blocks

def process_ans_blocks(ans_block):
    list_ans = []
    for item in ans_block:
        ans_block_img  = numpy.array(item[0])
        offset1 = ceil(ans_block_img.shape[0]/6)
        for i in range(6):
            box_img = numpy.array(ans_block_img[i*offset1:(i+1)*offset1,:])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box-14, :]
            offset2 = ceil(box_img.shape[0]/5)

            for j in range(5):
                list_ans.append(box_img[j*offset2:(j+1)*offset2, :])
    return list_ans


if __name__ == '__main__':
    img = cv.imread("./8f8fa123-b8c8-4fa0-aa63-b8750279e028.jpg")
    list_ans_boxs = crop_img(img)
    list_box = process_ans_blocks(list_ans_boxs)
    plt.imshow(list_box[0], interpolation='nearest')
    plt.show()
    print(list_box)
