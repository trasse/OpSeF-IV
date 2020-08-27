
# from https://github.com/mpicbg-csbd/stardist / 3_prediction (2D)
from __future__ import print_function, unicode_literals, absolute_import, division
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D, StarDist3D


# basic libs
import os
import shutil
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
import pickle
import networkx as nx
import readlif
from readlif.reader import LifFile

# skimage
import skimage
from skimage import transform, io, filters, measure, morphology,img_as_float
from skimage.color import label2rgb,gray2rgb
from skimage.filters import gaussian, rank, threshold_otsu
from skimage.io import imread, imsave
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, watershed

# scipy
from scipy.signal import medfilt
from scipy.ndimage import generate_binary_structure, binary_dilation

# cellpose
from cellpose import models as cp_models
from cellpose import utils as cp_utils
from cellpose import plot, transforms
from cellpose import plot, transforms
import mxnet as mx

# import from import_path
from Tools_002_3D import *
from UNet_CP01 import *
from Segmentation_Func_08_3E import *
from Pre_Post_Process002_3D import *
from N2V_DataGeneratorTR001 import *

# other
import pkg_resources
import keras

# for cluster analysis
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

# from noise to void (coming soon)
from n2v.models import N2VConfig, N2V
from n2v.utils.n2v_utils import manipulate_val_data



def split_tif_to_substacts(large_stack_fn, root, zstep):
    # splits a large tiffstack into multiple substacks
    large_stack = tif.imread(large_stack_fn)
    p,f = os.path.split(large_stack_fn)
    print(large_stack.shape)
    z,y,x = large_stack.shape
    mydtype = large_stack.dtype
    znow = 0
    while znow + zstep < z:
        substack = np.zeros((zstep,y,x), dtype=mydtype)
        for zsub in range (0,zstep):
            substack[zsub,:,:] = large_stack[znow+zsub,:,:]
            img_name = os.path.join(root, "tiff","SubZ_{}_{}_{}".format(str(znow).zfill(3),str(znow + zstep).zfill(3),f))
            tif.imsave(img_name, substack)
        znow += zstep
    return

def define_lif_pipeline(input_def):
    # defines a processing pipeline starting with a .lif file
    fpath_in_lif = glob("{}/*lif".format(input_def["root"]))[0] # use only one lif file prer folder!!!
    print("Analysing",fpath_in_lif)
    # load lif file
    lifobj = LifFile(fpath_in_lif)
    if input_def["split_z"]:
        input_def["input_type"] = ".tif" # further processing of the lif file is then based on the tiff folder!!!!
        new_folder = os.path.join(input_def["root"],"tiff")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print("Creating folder..." + new_folder)
        for ex_ch in input_def["export_multiple_ch"]:
            split_lif_z_to_tiff(lifobj,input_def["root"],input_def["z_step"],ex_ch,input_def["rigth_size"],input_def["mydtype"])
    return lifobj,input_def

def make_folder_structure(pc,input_def,run_def):
    # creates the folder structure
    # create tiff folder (in case needed)
    top_level = ["tiff","tiff_to_split"]
    for tl in top_level:
        new_folder = os.path.join(input_def["root"],tl)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print("Creating folder..." + new_folder)
    # create subfolder per run
    for sf in pc["sub_f"]:
        new_folder = os.path.join(input_def["root"],"Processed_{}".format(run_def["run_ID"]),sf)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print("Creating folder..." + new_folder)
    return

def preprocess_1_for_lif(lifobj,input_def,pc,run_def):
    # does stage1 of the preprocessing of data
    # (parameter free)
    for i in range(0,lifobj.num_images):
        img = lifobj.get_image(i)
        print(img.dims[0:2])
        if (img.name[0:4] in input_def["subset"] or input_def["subset"][0] == "All") and (img.dims[0:2] == input_def["rigth_size"]):
            print("Analysing..", img.name)
            x,y,z,t =(img.dims)
            img_load = np.zeros((x,y,z),dtype = input_def["mydtype"])
            for zp in range(0,z):
                im = np.asarray(img.get_frame(z=zp,t=0,c=input_def["export_single_ch"]))
                img_load[:,:,zp] = im
            preprocess_step1(img.name,img_load,input_def["root"],pc["sub_f"],run_def["run_ID"])
    return

def preprocess_2_for_lif(lifobj,input_def,pc,run_def):
    # does stage2 of the preprocessing of data
    # (according to the parameter defined in run_def)
    for jj,para in enumerate(run_def["pre_list"]):
        for i in range(0,lifobj.num_images):
            img = lifobj.get_image(i)
            if (img.name[0:4] in input_def["subset"] or input_def["subset"][0] == "All") and (img.dims[0:2] == input_def["rigth_size"]):
                print("Analysing..", img.name)
                x,y,z,t =(img.dims)
                img_input = np.zeros((x,y,z),dtype = input_def["mydtype"])
                for zp in range(0,z):
                    img_input[:,:,zp] = np.asarray(img.get_frame(z=zp,t=0,c=input_def["export_single_ch"]))
                preprocess_step2(img.name,img_input,para,input_def["root"],pc["sub_f"],run_def["run_ID"],jj)
    return

def preprocess_1_for_tif(fpath_list,input_def,pc,run_def):
    # does stage1 of the preprocessing of data
    # (parameter free)
    print("preprocess_1_for_tif")
    mem_save = True
    for fn in fpath_list:
        p,f = os.path.split(fn)
        if mem_save:
            img_in = tif.imread(fn).astype(np.uint8) # z,y,x
        else:
            img_in = tif.imread(fn)
            #print(img.shape)
        if len(img_in.shape) == 2:
            print("reshape")
            img_in = img_in[None,:,:]
            #img2 = np.reshape(img_in,(1,img_in.shape[0],img_in.shape[1]))
        else:
            pass
            #img2 = img_in
        #img = np.moveaxis(img2,0,2)
        img_in = np.moveaxis(img_in, 0, 2)
        #preprocess_step1(f.replace(".tif",""),img,input_def["root"],pc["sub_f"],run_def["run_ID"])
        print("preprocess")
        if mem_save:
            print("project")
            proj_img = np.sum(img_in, axis=2)
            del img_in
            preprocess_step1_mem_save(f.replace(".tif", ""), proj_img, input_def["root"], pc["sub_f"], run_def["run_ID"])
            del proj_img
        else:
            preprocess_step1(f.replace(".tif", ""), img_in, input_def["root"], pc["sub_f"], run_def["run_ID"])
            del img_in
    return

#def preprocess_1_for_3Dtif(fpath_list,input_def,pc,run_def):
#    # copies data for 3D processing
#    for fn in fpath_list:
#        p,f = os.path.split(fn)
#        img_in = tif.imread(fn) # z,y,x
#        fn_out = os.path.join(input_def["root"], "Processed_{}".format(run_def["run_ID"]), pc["sub_f"][1],f)
#        print("Saving...",fn_out)
#        tif.imsave(fn_out,img_in)
#    return

def preprocess_2_for_tif(fpath_list,input_def,pc,run_def):
    # does stage2 of the preprocessing of data
    # (parameter free)
    print("preprocess_2_for_tif")
    for jj,para in enumerate(run_def["pre_list"]):
        for fn in fpath_list:
            p,f = os.path.split(fn)
            img_in = tif.imread(fn)
            if len(img_in.shape) == 2:
                img2 = np.reshape(img_in,(1,img_in.shape[0],img_in.shape[1]))
            else:
                img2 = img_in
            img = np.moveaxis(img2,0,2)
            preprocess_step2(f.replace(".tif",""),img,para,input_def["root"],pc["sub_f"],run_def["run_ID"],jj)
            del img_in
    return

def preprocess_3D_for_tif(fpath_list,input_def,pc,run_def):
    # does stage2 of the preprocessing of data
    # (parameter free)
    for jj,para in enumerate(run_def["pre_list"]):
        for fn in fpath_list:
            p,f = os.path.split(fn)
            img_in = tif.imread(fn)
            preprocess_3D(f.replace(".tif",""),img_in,para,input_def["root"],pc["sub_f"],run_def["run_ID"],jj)
    return

def define_tif_pipeline(input_def,run_def,pc):
    # defines a processing pipeline starting with a .tiff file
    print("define_tif_pipeline")
    ##### stage 1 processing ##################
    if input_def["bin"]:  # search for input in "tiff_raw"
        if not input_def["toTiles"]:  # tile & bin pipelines are handled in toTiles
            Bin_image(input_def)

    if input_def["toTiles"]:  # search for input in "tiff_raw"
        if input_def["is3D"]:
            make_patches_3D(input_def)
        else:
            make_patches_2D(input_def)

    ############# coming soon #######################
    if input_def["n2v"]:  # search for input in "tiff_raw"
        pass
    if input_def["CARE"]:  # search for input in "tiff_raw"
        pass

    ###### stage 2 processing ###########################
    #### output of stage 1 might be input to stage 2 ####

    if input_def["split_z"]:
        search_path = os.path.join(input_def["root"], "tiff_to_split")
        tiff_to_split = glob("{}/*tif".format(search_path))
        for fn in tiff_to_split:
            split_tif_to_substacts(fn, input_def["root"], input_def["z_step"])

    fpath_list = get_file_list_tiff(input_def["root"], input_def["subset"])

    ################ IMPROVE LATER!!
    ################ these lines bypass most preprocessing for 3D data
    if input_def["seg_type"] == "3D":
        print("3D pipeline, only basic preprocessing")
        #for fp in fpath_list:
        #    fn = os.path.split(fp)[1]
        #    fpnew = os.path.join(input_def["root"], "Processed_{}".format(run_def["run_ID"]), pc["sub_f"][1],fn)
        #    shutil.copy(fp,fpnew)
    return fpath_list

def make_patches_2D(input_def):
    # makes patches in 3d
    search_here = os.path.join(input_def["root"],"tiff_raw_2D")
    datagen = N2V_DataGenerator2()
    fn_list, imgs = datagen.load_imgs_from_directory(directory = search_here, dims = "YX",names_back = True, to32bit = False)
    for i,img in enumerate(imgs):
        X = datagen.generate_patches(imgs[0], shape=input_def["patch_size"],augment = False, shuffle_patches = False)
        for p in range(0,X.shape[0]):
            patch_img = X[p,:,:,0]
            print(patch_img.dtype)
            if input_def["bin"]:
                ynew = int(patch_img.shape[0]//input_def["bin_factor"])
                xnew = int(patch_img.shape[1]//input_def["bin_factor"])
                bin_img =  cv2.resize(patch_img, dsize=(ynew,xnew), interpolation=cv2.INTER_LINEAR)
                img_name = os.path.join(input_def["root"],"tiff","PatchBIN{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                tif.imsave(img_name,bin_img)
            else:
                img_name = os.path.join(input_def["root"],"tiff","Patch{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                tif.imsave(img_name,patch_img)
    return

def make_patches_3D(input_def):
    # makes patches in 3d
    search_here = os.path.join(input_def["root"],"tiff_raw")
    datagen = N2V_DataGenerator2()
    fn_list, imgs = datagen.load_imgs_from_directory(directory = search_here, dims = "ZYX",names_back = True, to32bit = False)
    for i,img in enumerate(imgs):
        X = datagen.generate_patches(imgs[0], shape=input_def["patch_size"],augment = False, shuffle_patches = False)
        for p in range(0,X.shape[0]):
            patch_img = X[p,:,:,:,0]
            if input_def["bin"]:
                ynew = int(patch_img.shape[1]//input_def["bin_factor"])
                xnew = int(patch_img.shape[2]//input_def["bin_factor"])
                bin_img = np.zeros((patch_img.shape[0],ynew,xnew),dtype = patch_img.dtype)
                for znow in range(0,patch_img.shape[0]):
                    bin_img[znow,:,:] =  cv2.resize(patch_img[znow,:,:], dsize=(ynew,xnew), interpolation=cv2.INTER_LINEAR)
                if input_def["split_z"]:
                    img_name = os.path.join(input_def["root"],"tiff_to_split","PatchBIN{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                else:
                    img_name = os.path.join(input_def["root"],"tiff","PatchBIN{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                tif.imsave(img_name,bin_img)
            else:
                if input_def["split_z"]:
                    img_name = os.path.join(input_def["root"],"tiff_to_split","Patch{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                else:
                    img_name = os.path.join(input_def["root"],"tiff","Patch{}-{}".format(str(p).zfill(3),os.path.split(fn_list[i])[1]))
                tif.imsave(img_name,patch_img)
    return

def Bin_image(input_def):
    # Bins the Input Data
    search_here = os.path.join(input_def["root"],"tiff_raw")
    datagen = N2V_DataGenerator2()
    # load data
    if input_def["is3D"]:
        fn_list, imgs = datagen.load_imgs_from_directory(directory = search_here, dims = "ZYX",names_back = True, to32bit = False)
    else:
        fn_list, imgs = datagen.load_imgs_from_directory(directory = search_here, dims = "YX",names_back = True, to32bit = False)
    print(fn_list)
    for i,img in enumerate(imgs):
        if input_def["is3D"]:
            im = img[0,:,:,:,0]
            #print(im.shape)
            ynew = int(im.shape[1]//input_def["bin_factor"])
            xnew = int(im.shape[2]//input_def["bin_factor"])
            bin_img = np.zeros((im.shape[0],ynew,xnew),dtype = img.dtype)
            for znow in range(0,im.shape[0]):
                bin_img[znow,:,:] =  cv2.resize(im[znow,:,:], dsize=(ynew,xnew), interpolation=cv2.INTER_LINEAR)
            print(input_def["split_z"])
            if input_def["split_z"]:
                img_name = os.path.join(input_def["root"],"tiff_to_split","BIN{}".format(os.path.split(fn_list[i])[1]))
            else:
                print("Bug")
                img_name = os.path.join(input_def["root"],"tiff","BIN{}".format(os.path.split(fn_list[i])[1]))
            tif.imsave(img_name,bin_img)
        else:
            im = img[0,:,:,0]
            ynew = int(im.shape[0]//input_def["bin_factor"])
            xnew = int(im.shape[1]//input_def["bin_factor"])
            bin_img =  cv2.resize(im, dsize=(ynew,xnew), interpolation=cv2.INTER_LINEAR)
            img_name = os.path.join(input_def["root"],"tiff","BIN{}".format(os.path.split(fn_list[i])[1]))
            tif.imsave(img_name,bin_img)
    return

#####################################
#### new from Opsef 0.1_47_m8

def export_second_channel_for_mask(lifobj,pc,input_def,run_def):
    # export second channel to create mask or to qualify
    if input_def["input_type"] == ".tif":
        print("Export from tiff")
        export_channel_from_tif2(input_def["root"],pc["sub_f"],run_def["run_ID"],input_def["post_subset"]) # test, revert to export_channel_from_tif if it fails
    if input_def["input_type"] == ".lif":
        print("Export from lif")
        export_channel_from_lif(lifobj,input_def["root"],input_def["post_export_single_ch"],input_def["rigth_size"],input_def["mydtype"],run_def["run_ID"],pc["sub_f"])
    exported_file_list = glob("{}/*tif".format(os.path.join(input_def["root"],"Processed_{}".format(run_def["run_ID"]),pc["sub_f"][5])))
    return exported_file_list

def make_mask_to_img_dic(mask_files, pc, input_def, run_def, which_subfolder = 0, channelID = 0):
    # makes dics that define the mask to intensity image pair
    if pc["naming_scheme"] == "Simple":
        MD = [6]  # where to split the string of the mask filename (elements are str.split("_"))
        CQ = ["Sum"]  # changes to convert mas filename to base image
        # filename that will be used for quantification
        mask_to_img_dic = make_pair_dic_simple(mask_files, MD, CQ, input_def["root"], run_def["run_ID"], pc["sub_f"],which_subfolder)
        CQ = ["8bitSum"]
        mask_to_8bitimg_dic = make_pair_dic_simple(mask_files, MD, CQ, input_def["root"], run_def["run_ID"],
                                                   pc["sub_f"],which_subfolder)

    if pc["naming_scheme"] == "Export_ZSplit":
        MD = [6,10,11]  # where to split the string of the mask filename  (elements are str.split("_"))
        CQ = ["Sum", str(channelID).zfill(3)]  # changes to convert mas filename to base image
        # filename that will be used for quantification
        mask_to_img_dic = make_pair_dic_Export_ZSplit(mask_files, MD, CQ, input_def["root"], run_def["run_ID"],
                                                      pc["sub_f"],which_subfolder)
        CQ = ["8bitSum", str(channelID).zfill(3)]
        mask_to_8bitimg_dic = make_pair_dic_simple(mask_files, MD, CQ, input_def["root"], run_def["run_ID"],
                                                   pc["sub_f"],which_subfolder)
    return mask_to_img_dic, mask_to_8bitimg_dic