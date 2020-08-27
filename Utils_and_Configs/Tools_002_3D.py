
# skimage
import skimage
from skimage import transform, io, filters, measure, morphology,img_as_float
from skimage.color import label2rgb,gray2rgb
from skimage.filters import gaussian, rank, threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import disk, watershed

# scipy
from scipy.signal import medfilt
from scipy.ndimage import generate_binary_structure, binary_dilation

# basic libs

import os
import sys
import time
import inspect
from glob import glob

import tifffile as tif

import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections

import networkx as nx


def ski_uniform_filter(img,myfilter,kernel):
    # applies a filter to an image
    # skiimage has different ways to define the kernel
    # this a function compensates for it
    if myfilter=="Median":
        kernel2d = np.ones((kernel,kernel),dtype = bool)
        result = skimage.filters.median(img, selem=kernel2d)
    elif myfilter=="Gaussian":
        result = skimage.filters.gaussian(img, kernel)
    elif myfilter=="Mean":
        selem = disk(kernel)
        result = rank.mean(img, selem=selem)
    elif myfilter=="None":
        result = img
    return result


def cv_binary_processor_plus(im, todo):
    # does a series of binary operations on an image
    # input is a list that provides the operation, kernel size and the number of iterations.
    # for theory of binary processing in open CV see: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    # se 0 creates a rectangular structuring element, se = 1 an eliptical
    # handels also distance transforms and thresholds

    # convert binary image for use in open cv
    img = np.zeros(im.shape, dtype=np.uint8)
    img[im] = 255  # open CV expects this for binary images

    for step in todo:
        if step[4] == "Morphology":
            op = step[0]
            ke = np.ones((int(step[1]), int(step[1])), np.uint8)
            it = int(step[2])
            se = int(step[3])
            if se == 0:
                ke = np.ones((int(step[1]), int(step[1])), np.uint8)
            else:
                ke = cv.getStructuringElement(cv.MORPH_ELLIPSE, ((int(step[1])), (int(step[1]))))
            # process
            if op == "erode":
                out = cv.erode(img, ke, iterations=it)
            if op == "dilate":
                out = cv.dilate(img, ke, iterations=it)
            if op == "close":
                out = cv.morphologyEx(img, cv.MORPH_CLOSE, ke, iterations=it)
            if op == "open":
                out = cv.morphologyEx(img, cv.MORPH_OPEN, ke, iterations=it)
            if op == "gradient":
                out = cv.morphologyEx(img, cv.MORPH_GRADIENT, ke, iterations=it)

        elif step[4] == "Various":
            if step[0] == "distance":
                # see https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
                # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042
                # we use always DIST_L2 (Euclidean distance)
                # step[1] = radius
                # step[2] = fold of maximum distance
                dist_transform = cv.distanceTransform(img, cv.DIST_L2, step[1])
                bin_out = np.zeros(im.shape, dtype=bool)
                bin_out[np.where(dist_transform > step[2] * dist_transform.max())] = 1
                out = np.zeros(im.shape, dtype=np.uint8)
                out[bin_out] = 255
        img = out
    img_out = np.zeros(im.shape, dtype=bool)
    img_out[np.where(out > 0)] = 1
    return img_out


def get_binary(myimg,para,rd):
    # creates a mask from a second channel

    pre,o_thres_f,binary_filter = para
    # filter
    img_pp = ski_uniform_filter(myimg,pre[0],pre[1])
    # thresholding
    try:
        thres = threshold_otsu(img_pp)
        print(thres)
    except: # if otsu fails the threshold is set to the max value
        thres = np.iinfo(img_pp.dtype).max
        print("Warning, Otsu failed!! ")
    binary_img = np.zeros(myimg.shape, dtype=bool)
    binary_img[np.where(img_pp > (thres * o_thres_f[0]))] = 1
    # binary filtering
    bin_pp = cv_binary_processor_plus(binary_img,binary_filter)
    return bin_pp

def stack_calculator(stack_a,stack_b,pv):
    # does image processing that depends on two stacks
    operation, parameter = pv
    res_img = np.zeros(stack_a.shape, dtype=stack_a.dtype)
    # Mask
    ##############################################################
    if operation == "mask_0":
        res_img = np.zeros(stack_a.shape, dtype=stack_a.dtype)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_b == 0)] = 1
        res_img = stack_a.copy()
        res_img[mask] = 0
    if operation == "mask_non_0":
        res_img = np.zeros(stack_a.shape, dtype=stack_a.dtype)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_b > 0)] = 1
        res_img = stack_a.copy()
        res_img[mask] = 0
    # Binary Logic
    ################################################################
    if operation == "or": # returns pixel that are non-zero either array (ideal on bool)
        res_img = np.zeros(stack_a.shape, dtype=bool)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where((stack_a > 0) | (stack_b > 0))] = 1
        res_img[mask] = 1
    if operation == "or0255": # returns pixel that are non-zero either array (ideal on bool)
        res_img = np.zeros(stack_a.shape, dtype=np.uint8)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where((stack_a > 0) | (stack_b > 0))] = 1
        res_img[mask] = 255
    if operation == "and": # returns pixel that are non-zero in both arrays (returns TRUE/FALSE array)
        res_img = np.zeros(stack_a.shape, dtype=bool)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where((stack_a > 0) & (stack_b > 0))] = 1
        res_img[mask] = 1
    if operation == "and0255": # returns pixel that are non-zero in both arrays (returns 0/255 array)
        res_img = np.zeros(stack_a.shape, dtype=np.uint8)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where((stack_a > 0) & (stack_b > 0))] = 1
        res_img[mask] = 255
    if operation == "A_not_B": # returns pixel that are non-zero in A and 0 in B
        res_img = np.zeros(stack_a.shape, dtype=bool)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_a > 0)] = 1
        res_img[mask] = 1
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_b > 0)] = 1
        res_img[mask] = 0
    if operation == "A_not_B0255": # returns pixel that are non-zero in A and 0 in B
        res_img = np.zeros(stack_a.shape, dtype=np.uint8)
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_a > 0)] = 1
        res_img[mask] = 255
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_b > 0)] = 1
        res_img[mask] = 0
    # Math
    ##################################################################
    if operation == "A_minus_Bx": # substracts B * a constant from A (e.g. to correct channel crosstalk)
        x = parameter[0]
        res_img = stack_a - stack_b * x
        mask = np.zeros(stack_b.shape, dtype=bool)
        mask[np.where(stack_a < stack_b * x)] = 1
        res_img[mask] = 0 # clip negative results
    if operation == "Ratio_A_B": # returns the ratio of both arrays
        np.seterr(all="ignore")
        x = parameter[0]
        res_img = np.divide(stack_a,stack_b)
        np.seterr(all="warn")
    if operation == "both_larger": # returns pixel in A,B that are above the thresholds TA, TB respectively
        TA = parameter[0]
        TB = parameter[1]
        res_img = np.zeros(stack_a.shape, dtype=bool)
        mask = np.zeros(stack_a.shape, dtype=bool)
        mask[np.where((stack_a > TA) & (stack_b > TB))] = 1
        res_img[mask] = 1
    if operation == "both_within": # returns pixel in A,B that are within a range in both arrays
        converted_para = []
        # determine lower and upper thresholds
        for i,para in enumerate(parameter):
            method,num = para
            if method == "fixed":
                converted_para.append(num)
            elif method == "perc":
                if i in (0,1):
                    converted_para.append(np.percentile(stack_a,num))
                else:
                    converted_para.append(np.percentile(stack_b,num))
            else:
                print("Error, define Threshold as fixed or percentile!")
                return
        TA1,TA2,TB1,TB2 = converted_para
        res_img = np.zeros(stack_a.shape, dtype=bool)
        mask = np.zeros(stack_a.shape, dtype=bool)
        mask[np.where((stack_a > TA1) & (stack_a < TA2) & (stack_b > TB1) & (stack_b < TB2) )] = 1
        res_img[mask] = 1
    return res_img

def draw_outline(img,mask):
    # shows outline of a mask in red
    result = img.copy()
    mask_bin = np.zeros(mask.shape, dtype = bool)
    mask_bin[np.where(mask > 0)] = 1
    mask_bin = cv_binary_processor_plus(mask_bin,[["dilate",3,1,1,"Morphology"]]) # erode
    core = get_core_img(mask)
    mask_bin[core] = 0 # delete center
    red = np.zeros(mask.shape, dtype = np.uint8)
    red[:,:] = result[:,:,0]
    red[mask_bin] = 255
    result[:,:,0] = red[:,:]
    print(result.shape)
    return result


def get_core_img(img):
    # gets core of an labelede image
    region_ids = np.unique(img)[1:]
    mask_return = np.zeros(img.shape, dtype = bool)
    for ri in region_ids:
        mask = np.zeros(img.shape, dtype = bool)
        mask[np.where(img == ri)] = 1
        mask = cv_binary_processor_plus(mask,[["erode",3,1,1,"Morphology"]])
        mask_return[mask] = 1
    return mask_return

def clahe_augment2(im,tg,cl):
# applies the open CV implementation of Contrast-limited adaptive histogram equalization (CLAHE)
# see https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe_low = cv.createCLAHE(clipLimit=cl, tileGridSize=tg)
    img_low = clahe_low.apply(im)
    return img_low

############################################################################
### new from Opsef_0_1_34_t2 -> other functions


def get_file_list_tiff(root,sset):
    # finds tiff files to be processed and returns them
    fpath_in_tiff = glob("{}/{}/*tif".format(root,"tiff"))
    print(fpath_in_tiff)
    if sset[0] == "All":
        fpath_list = fpath_in_tiff
    else:
        fpath_list = [f for f in fpath_in_tiff if len([x for x in sset if x in os.path.split(f)[1]]) > 0]
    # checks is any of the wanted substrings in my filename?
    print("Analysing",fpath_list)
    return fpath_list

