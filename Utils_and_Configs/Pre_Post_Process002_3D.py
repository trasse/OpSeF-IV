# basic libs

import os
import sys
import time
import inspect
from glob import glob

import tifffile as tif

import cv2 as cv
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import math

import networkx as nx

# skimage
import skimage
from skimage import transform, io, filters, measure, morphology,img_as_float
from skimage.color import label2rgb,gray2rgb
from skimage.filters import gaussian, rank, threshold_otsu,roberts,sobel
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, watershed

from Tools_002_3D import *

# from functions for preprocessing

def preprocess_step1(imname,im,root,sub_folder,runID):
    # does the preprocessing
    # (parameter free)
    # make sum projection
    print("project")
    proj_img = np.sum(im,axis=2)
    del im
    print("8bit")
    proj_img_8bit = (255 * proj_img / np.max(proj_img)).astype(np.uint8)
    # save 32 bit raw
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[0],"Sum_{}.tif".format(imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img)
    del proj_img
    # save 8 bit raw
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[0],"8bitSum_{}.tif".format(imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img_8bit)
    del proj_img_8bit
    return

def preprocess_step1_mem_save(imname,proj_img,root,sub_folder,runID):
    # does the preprocessing
    # (parameter free)
    # make sum projection
    print("8bit")
    proj_img_8bit = (255 * proj_img / np.max(proj_img)).astype(np.uint8)
    # save 32 bit raw
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[0],"Sum_{}.tif".format(imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img)
    del proj_img
    # save 8 bit raw
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[0],"8bitSum_{}.tif".format(imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img_8bit)
    del proj_img_8bit
    return

def preprocess_3D(imname,im_in,pa,root,sub_folder,runID,jj):
    # does the preprocessing based on the predefined parameter
    # the parameter ID will be encoded in the filename
    myfilter,mykernel,subs,resize,apply_clahe,cl_prm,enhance_edge,invert = pa
    x,y,z = (im_in.shape)
    img_clean = np.zeros((x,y,z),dtype = im_in.dtype)
    # filter images (uses: myfilter,mykernel,subs)
    for zp in range(0,z):
        plane = im_in[:,:,zp]
        im_clean = ski_uniform_filter(plane.copy(),myfilter,mykernel)
        mask = np.zeros(im_clean.shape,dtype = bool)
        mask[np.where(im_clean < subs)] = 1
        im_clean = im_clean - subs
        im_clean[mask] = 0
        img_clean[:,:,zp] = im_clean
    # resize (uses resize)
    if resize != (1,1,1):
        target_shape = [round(x * resize[ii]) for ii,x in enumerate(img_clean.shape)]
        img_out = skimage.transform.resize(img_clean, target_shape, preserve_range=True, anti_aliasing=True).astype(img_clean.dtype)
    else:
        img_out = img_clean
    # adapt histogram (apply_clahe, cl_prm)
    if apply_clahe:
        print("Contrast Limited Adaptive Histogram Equalization is not supported for 3D images")
    # enhance edges
    if enhance_edge != "no":
        print("Edge enhancement is not supported for 3D images")
        print("and does typically not result in better results")
    # invert
    if invert:
        print("Invert is not supported for 3D images")
    # Save image
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[1],"Input_{}_{}.tif".format(str(jj).zfill(3),imname))
    print("Saving..",img_name)
    tif.imsave(img_name,img_out)
    return

def preprocess_step2(imname,im_in,pa,root,sub_folder,runID,jj):
    # does the preprocessing based on the predefined parameter
    # the parameter ID will be encoded in the filename
    mem_save = True # pass on to function!!!
    myfilter,mykernel,subs,myproj,apply_clahe,cl_prm,enhance_edge,invert = pa
    x,y,z = (im_in.shape)
    print("Create Clean")
    img_clean = np.zeros((x,y,z),dtype = im_in.dtype)
    # process images
    for zp in range(0,z):
        plane = im_in[:,:,zp]
        im_clean = ski_uniform_filter(plane.copy(),myfilter,mykernel)
        mask = np.zeros(im_clean.shape,dtype = bool)
        mask[np.where(im_clean < subs)] = 1
        im_clean = im_clean - subs
        im_clean[mask] = 0
        img_clean[:,:,zp] = im_clean
        # make projection
    if myproj == "Sum":
        proj_img_clean = np.sum(img_clean,axis=2)
        proj_img_8bit_clean = ((255 * proj_img_clean) / np.max(proj_img_clean)).astype(np.uint8)
    elif myproj == "Max":
        proj_img_clean = np.amax(img_clean,axis=2)
        proj_img_8bit_clean = proj_img_clean.astype(np.uint8)
    elif myproj == "Median":
        proj_img_clean = np.median(img_clean, axis=2)
        proj_img_8bit_clean = proj_img_clean.astype(np.uint8)
    if enhance_edge != "no":
        if enhance_edge == "roberts":
            edge_enhanced = filters.roberts(proj_img_8bit_clean) # float64 result !!
        if enhance_edge == "sobel":
            edge_enhanced  = filters.sobel(proj_img_8bit_clean) # float64 result !!
        proj_img_8bit_clean = ((255 * edge_enhanced) / np.max(edge_enhanced)).astype(np.uint8)
    if invert:
        proj_img_8bit_clean = 255 - proj_img_8bit_clean.copy()
    if apply_clahe:
        proj_img_8bit_clean = clahe_augment2(proj_img_8bit_clean,cl_prm[0],cl_prm[1])
        # save 32 bit clean
    print("Save 32")
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[0],"CleanSum_{}_{}.tif".format(str(jj).zfill(3),imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img_clean)
    del proj_img_clean
        # save 8 bit clean
    print("Save 8")
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[1],"Input_{}_{}.tif".format(str(jj).zfill(3),imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img_8bit_clean)
    # save as RGB
    print("Save RGB")
    if mem_save:
        print("No RGB")
    else:
        print("RGB")
        RGB = np.zeros((proj_img_8bit_clean.shape[0],proj_img_8bit_clean.shape[1],3),dtype = proj_img_8bit_clean.dtype)
        for zz in (0,1,2):
            RGB[:,:,zz] = proj_img_8bit_clean[:,:]
        img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[1],"RGBInput_{}_{}.tif".format(str(jj).zfill(3),imname))
        tif.imsave(img_name,RGB)
    return

def split_lif_z_to_tiff(lifobj, root, zstep, mych, rightsize,mydtype):
    # splits a lif file into multiple files
    for i in range(0, lifobj.num_images):
        img = lifobj.get_image(i)
        if img.dims[0:2] == rightsize:
            print("Exporting..", img.name)
            x, y, z, t = (img.dims)
            img_load = np.zeros((z, 1, x, y), dtype=mydtype)
            for zp in range(0, z):
                im = np.asarray(img.get_frame(z=zp, t=0, c=mych))
                img_load[zp, 0, :, :] = im
            zsub = 0
            print(img_load.shape)
            while zsub + zstep < z:
                substack = img_load[zsub:(zsub + zstep), 0, :, :]
                # print(substack.shape)
                img_name = os.path.join(root, "tiff",
                                        "SubZ_{}_{}_Ch_{}_{}.tif".format(str(zsub).zfill(3), str(zsub + zstep).zfill(3),
                                                                         str(mych).zfill(3), img.name))
                tif.imsave(img_name, substack)
                zsub = zsub + zstep
    return

