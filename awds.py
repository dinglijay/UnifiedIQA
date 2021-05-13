import cv2
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
dy = dx.T

def gradMag(gray):
    ref_dx = cv2.filter2D(gray, cv2.CV_32F, dx)
    ref_dy = cv2.filter2D(gray, cv2.CV_32F, dy)
    grad = cv2.sqrt(cv2.pow(ref_dx, 2) + cv2.pow(ref_dy, 2))
    return grad

def AWDS(ref_file, dis_file):
    ref_img = cv2.imread(ref_file,cv2.IMREAD_GRAYSCALE)
    dis_img = cv2.imread(dis_file,cv2.IMREAD_GRAYSCALE)

    size, sigma = (25, 25), 0
    c, a = 0.0025 * 65535, 0.7
    ref_GM_0 = gradMag(ref_img)
    dis_GM_0 = gradMag(dis_img)
    mu1 = cv2.GaussianBlur(ref_GM_0, size, sigma)
    mu2 = cv2.GaussianBlur(dis_GM_0, size, sigma)
    weight_map = cv2.max(mu1,mu2)
    qualityMap = ((2 - a) * ref_GM_0 * dis_GM_0 + c) / \
                  (cv2.pow(ref_GM_0, 2) + cv2.pow(dis_GM_0, 2) - a * ref_GM_0 * dis_GM_0 + c)
    score_fine = cv2.sumElems(qualityMap * weight_map)[0] / cv2.sumElems(weight_map)[0]

    c, a = 0.0025 * 65535, -10
    ref_GM_1 = gradMag(cv2.blur(ref_img, (2, 2))[::2,::2])
    dis_GM_1 = gradMag(cv2.blur(dis_img, (2, 2))[::2,::2])
    mu1 = cv2.GaussianBlur(ref_GM_1, size, sigma)
    mu2 = cv2.GaussianBlur(dis_GM_1, size, sigma)
    weight_map1 = cv2.max(mu1,mu2)
    qualityMap1 = ((2 - a) * ref_GM_1 * dis_GM_1 + c) / \
                  (cv2.pow(ref_GM_1, 2) + cv2.pow(dis_GM_1, 2) - a * ref_GM_1 * dis_GM_1 + c)
    score_coarse = cv2.sumElems(qualityMap1 * weight_map1)[0] / cv2.sumElems(weight_map1)[0]

    def getGDoG(gb_size=(5, 5)):
        grad0 = ref_GM_0
        grad1 = gradMag(cv2.GaussianBlur(ref_img, (5, 5), 0))
        c, a = 0.0025 * 65535, -10
        GDoG = ((2 - a) * grad0 * grad1 + c) / (cv2.pow(grad0, 2) + cv2.pow(grad1, 2) - a * grad0 * grad1 + c)
        weight_map = cv2.max(cv2.GaussianBlur(grad0, (5, 5), 0),cv2.GaussianBlur(grad1, (5, 5), 0))
        GDoG = cv2.sumElems(GDoG * weight_map)[0] / cv2.sumElems(weight_map)[0]
        norm_GDoG = sigmoid(2*(97.49502237 * mean - 90.52996552))
        return norm_GDoG

    GDoG = getGDoG(ref_img)
    mean = score_fine * (1 - GDoG) + GDoG * (score_coarse**4)

    return mean
