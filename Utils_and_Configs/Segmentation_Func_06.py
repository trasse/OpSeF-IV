from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import os
from csbdeep.utils import Path, normalize
from stardist import random_label_cmap, _draw_polygons
from stardist.models import StarDist2D
import tifffile as tif
import pkg_resources
import sys
import time
import skimage.io
import os.path
import numpy as np
from skimage import transform
import keras
from skimage.filters import gaussian
from scipy.signal import medfilt
import cv2
from skimage import filters
from skimage.morphology import watershed
import glob
from skimage.measure import label, regionprops, regionprops_table
import math
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from cellpose import models as cp_models
from cellpose import utils as cp_utils
from cellpose import plot, transforms
from cellpose import plot, transforms
import mxnet as mx

from Tools_002 import *
from UNet_CP01 import *
from Pre_Post_Process002 import *

def Run_StarDist2D(img,mdls,which_model):
    # runs a Stardist 2D model
    axis_norm = (0,1)   # normalize channels independently
    im = normalize(img, 1,99.8, axis=axis_norm)
    model = mdls[which_model]
    labels, details = model.predict_instances(im)
    return labels

def UNet_Watershed(im, mdls, which_model, m_kernel=2, de_kernel=2, de_it=2, dist_thr=0.5, show_all=False):
    # get probabilities for cell, boundaries and background from a U-Net
    # then does watershed as described in
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    # filter predictions

    p = unet_classify(mdls[which_model], im)

    kernel_size = m_kernel
    bckg = skimage.filters.gaussian(p[:, :, 0], kernel_size)
    cell = skimage.filters.gaussian(p[:, :, 1], kernel_size)
    bounda = skimage.filters.gaussian(p[:, :, 0], kernel_size)

    # get new cell mask
    cell_msk = np.zeros(cell.shape, dtype=np.uint8)
    cm = np.logical_and(np.greater(cell, bckg), np.greater(cell, bounda))
    bm = np.logical_and(np.greater(bounda, bckg), np.greater(bounda, cell))
    cell_msk[cm] = 255  # open CV expects this for binary images

    # noise removal
    dkernel = np.ones((de_kernel, de_kernel), np.uint8)
    opening = cv2.morphologyEx(cell_msk, cv2.MORPH_OPEN, dkernel, iterations=de_it)

    # Finding sure cell area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, dist_thr * dist_transform.max(), 255, 0)
    sure_fg2 = sure_fg.astype(np.uint8)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg2)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    unknown = np.zeros(cell.shape, dtype=bool)
    unknown[cm] = 1  # former cell area
    unknown[bm] = 1  # former boundary
    unknown[np.where(sure_fg > 0)] = 0  # minus sure cell area
    markers[unknown] = 0

    # define background
    bckgm = np.logical_and(np.greater(bckg, bounda), np.greater(bckg, cell))

    # calcumate gradient image for watershed
    cell[bckgm] = 0
    cell = cell * 255
    cell[:, :] = 255 - cell[:, :]

    # do watershed
    result_img = watershed(cell, markers)

    # print results
    if show_all:
        pass
    # return result_img, dist_transform, sure_fg
    return result_img

############################################################################
### new from Opsef_0_1_34_t2 -> functions segmentation
def initialize_model(init_model,model_type,which_model):
    # initialize models
    # returns a dictonary of models that can be called
    mds = {}
    # UNet Cell Profiler
    if model_type == "Cellprofiler_UNet":
        option_dict_conv,option_dict_bn = init_model["UNetSettings"]
        os.environ["KERAS_BACKEND"] =  "tensorflow"
        if which_model == "UNet_CP001":
            model = unet_initialize(init_model["UNetShape"],option_dict_conv,option_dict_bn,init_model["UNet_model_file_CP01"], automated_shape_adjustment=True)
            mds["UNet_CP001"] = model
    # Stardist
    elif model_type == "StarDist":
        if which_model == "SD_2D_dsb2018":
            model = StarDist2D(None, name='2D_dsb2018', basedir = init_model["basedir_StarDist"])
            mds["SD_2D_dsb2018"] = model
    # Cellpose
    elif model_type == "Cellpose":
        # check if GPU working, and if so use it
        use_gpu = cp_utils.use_gpu()
        if use_gpu:
            device = mx.gpu()
            print("GPU found")
        else:
            device = mx.cpu()
            print("CPU only")
        if which_model == "CP_nuclei":
            model = cp_models.Cellpose(device, model_type="nuclei")
            mds["CP_nuclei"] = model
        if which_model == "CP_cyto":
            model = cp_models.Cellpose(device, model_type="cyto")
            mds["CP_cyto"] = model
    print("Model_keys", mds.keys())
    return mds


def run_batch_cellpose(run_sub_id,mdls,which_model,path,img_batch,img_display,batch_fn_core,rl,chs,obj_filter,sub_folder,runID):
    # runs a batch of files in cellpose
    # return their cell number in a dic [processing_parameter][cell_name]
    results_dic = {}
    for rescale_it in rl:
    # compute result
        results_dic["{}_CP_".format(str(run_sub_id).zfill(3))+str(rescale_it)] = {}
        masks, flows, styles, diams = mdls[which_model].eval(img_batch, rescale=rescale_it, channels=chs)
    # save results
        for mi,mask in enumerate(masks):
        # save mask
            if img_display: # if not empty, use it
                mask = filter_results_rp(mask,img_display[mi][:,:,0].copy(),obj_filter)
            else:
                mask = filter_results_rp(mask,img_batch[mi][:,:,0].copy(),obj_filter)
            results_dic["{}_CP_".format(str(run_sub_id).zfill(3))+str(rescale_it)][batch_fn_core[mi]] = len(np.unique(mask)) -1 # number of cells
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[2],"{}_CP_Mask_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            tif.imsave(img_name,mask)
        # create outline
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[3],"{}_CP_MaskOutline_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            if img_display: # if not empty, use it
                overlay = draw_outline(img_display[mi],mask)
            else:
                overlay = draw_outline(img_batch[mi],mask)
            tif.imsave(img_name,overlay)
    return results_dic

def run_batch_stardist(run_sub_id,mdls,which_model,path,img_batch,img_display,batch_fn_core,obj_filter,sub_folder,runID,rl = [0]):
    # runs a batch of files in cellpose
    # return their cell number in a dic [processing_parameter][cell_name]
    results_dic = {}
    # compute result
    for rescale_it in rl:
        results_dic["{}_SD_".format(str(run_sub_id).zfill(3))+str(rescale_it)] = {}
        masks = []
        for img in img_batch:
            masks.append(Run_StarDist2D(img[:,:,0].copy(),mdls,which_model))
    # save results
        for mi,mask in enumerate(masks):
        # save mask
            if img_display: # if not empty, use it
                mask = filter_results_rp(mask,img_display[mi][:,:,0].copy(),obj_filter)
            else:
                mask = filter_results_rp(mask,img_batch[mi][:,:,0].copy(),obj_filter)
            results_dic["{}_SD_".format(str(run_sub_id).zfill(3))+str(rescale_it)][batch_fn_core[mi]] = len(np.unique(mask)) -1 # number of cells
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[2],"{}_SD_Mask_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            tif.imsave(img_name,mask)
        # create outline
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[3],"{}_SD_MaskOutline_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            if img_display: # if not empty, use it
                overlay = draw_outline(img_display[mi],mask)
            else:
                overlay = draw_outline(img_batch[mi],mask)
            tif.imsave(img_name,overlay)
    return results_dic

def run_batch_Unet(run_sub_id,mdls,which_model,path,img_batch,img_display,batch_fn_core,obj_filter,sub_folder,runID,rl = [0]):
    # runs a batch of files in cellpose
    # return their cell number in a dic [processing_parameter][cell_name]
    results_dic = {}
    # compute result
    for rescale_it in rl:
        results_dic["{}_UN_".format(str(run_sub_id).zfill(3))+str(rescale_it)] = {}
        masks = []
        for img in img_batch:
            masks.append(UNet_Watershed(img[:,:,0].copy(),mdls,which_model))
    # save results
        for mi,mask in enumerate(masks):
        # save mask
            if img_display: # if not empty, use it
                mask = filter_results_rp(mask,img_display[mi][:,:,0].copy(),obj_filter)
            else:
                mask = filter_results_rp(mask,img_batch[mi][:,:,0].copy(),obj_filter)
            results_dic["{}_UN_".format(str(run_sub_id).zfill(3))+str(rescale_it)][batch_fn_core[mi]] = len(np.unique(mask)) -1 # number of cells
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[2],"{}_UN_Mask_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            tif.imsave(img_name,mask)
        # create outline
            img_name = os.path.join(path,"Processed_{}".format(runID),sub_folder[3],"{}_UN_MaskOutline_{}_{}".format(str(run_sub_id).zfill(3),rescale_it,batch_fn_core[mi]))
            print("Saving..",img_name)
            if img_display: # if not empty, use it
                overlay = draw_outline(img_display[mi],mask)
            else:
                overlay = draw_outline(img_batch[mi],mask)
            tif.imsave(img_name,overlay)
            tif.imsave(img_name,overlay)
    return results_dic

# functions postprocessing
def filter_results_rp(label_img,fl_img,para):
    # this function filters the results by minimal and maximal size of cells
    all_cells = np.zeros(label_img.shape,dtype=bool)
    positive = np.where(label_img > 0)
    all_cells[positive] = 1
    # get region properties
    props = regionprops(label_img,fl_img)
    regions_ok = []
    # filter regions
    for region in props:
        if region.label > 0:
            circularity = (region.equivalent_diameter * math.pi) / region.perimeter
            sum_intensity = region.mean_intensity * region.area
            if (region.area > para["area"][0]) and (region.area < para["area"][1]) and \
            (region.perimeter > para["perimeter"][0]) and (region.perimeter < para["perimeter"][1]) and \
            (region.mean_intensity > para["mean_intensity"][0]) and (region.mean_intensity < para["mean_intensity"][1]) and \
            (region.eccentricity > para["eccentricity"][0]) and (region.eccentricity < para["eccentricity"][1]) and \
            (sum_intensity > para["sum_intensity"][0]) and (sum_intensity < para["sum_intensity"][1]) and \
            (circularity > para["circularity"][0]) and (circularity < para["circularity"][1]):
                pass
            else:
                regions_ok.append(region.label)
    # delete regions that do not match:
    for mv in regions_ok:
        if mv > 0:
            mask_this_cell = np.zeros(label_img.shape,dtype=bool)
            mtc = np.where(label_img == mv)
            mask_this_cell[mtc] = 1
            all_cells[mask_this_cell] = 0
    # create deletion map
    inverse = np.where(all_cells == 0)
    label_img[inverse] = 0
    return label_img

# functions for anlysis (added after version OpseF_0_1_40_m3)
def export_channel_from_lif(lifobj, root, mych, rightsize,mydtype,run_ID,sub_f):
    '''
    Images from additional channel will follow this naming scheme:
    Main_Folder/AdditionalChannel/
        - Sum_$FN
        - sum projection of original file as dtype=np.uint64 suited for quantification
         (but can not be opened in ImageJ)
    '''
    for i in range(0, lifobj.num_images):
        img = lifobj.get_image(i)
        if img.dims[0:2] == rightsize:
            print("Exporting..", img.name)
            x,y,z,t =(img.dims)
            img_load = np.zeros((x,y,z),dtype = mydtype)
            for zp in range(0, z):
                im = np.asarray(img.get_frame(z=zp,t=0,c=mych))
                img_load[:,:,zp] = im
            img_save = np.sum(img_load,axis=2)
            img_name = os.path.join(root,"Processed_{}".format(run_ID),sub_f[5],"Sum_Ch_{}_{}.tif".format(str(mych).zfill(3), img.name))
            tif.imsave(img_name, img_save)
    return

def export_channel_from_tif(root,sub_f,run_ID,subset):
    '''
    Images from additional channel will follow this naming scheme:
    Main_Folder/AdditionalChannel/
        - Sum_$FN
        - sum projection of original file as dtype=np.uint64 suited for quantification
         (but can not be opened in ImageJ)
    '''
    fpath_list = get_file_list_tiff(root,subset)
    for fn in fpath_list:
        p,f = os.path.split(fn)
        img = tif.imread(fn)
        img = np.moveaxis(img,0,2)
        export_channel(f.replace(".tif",""),img,root,sub_f,run_ID)
    return

def export_channel(imname,im,root,sub_folder,runID):
    # does the preprocessing
    # (parameter free)
    # make sum projection
    proj_img = np.sum(im,axis=2)
    # save 32 bit raw
    img_name = os.path.join(root,"Processed_{}".format(runID),sub_folder[5],"Sum_{}.tif".format(imname))
    print("Saving..",img_name)
    tif.imsave(img_name,proj_img)
    return

# naming scheme DemoOpSef 3
def make_pair_dic_Export_ZSplit(masks_input_files,MD,CQ,root,run_ID,sub_f,which_subfolder = 5):
    # makes dic of filepath for mask to base image that will be used for quantification
    mask_to_img = {}
    for mask_fn in masks_input_files:
        split_fn = os.path.split(mask_fn)[1].split("_")
        new_fn = "_".join([CQ[0]] + split_fn[MD[0]:MD[1]] + [CQ[1]] + split_fn[MD[2]:])
        fp_new = os.path.join(root,"Processed_{}".format(run_ID),sub_f[which_subfolder],new_fn)
        mask_to_img[mask_fn] = fp_new
    return mask_to_img

def results_to_csv(mask_to_img_dic,get_property,root,sub_f,run_ID,output_folder,tag,subset):
    '''
    Here the results are extracted and saved as csv file.
    The naming scheme of the folder Basic_Quantification is as follows:
    Combined_Object_$Data_Analysis_ID_$Search_Term_you_used_to_filter_results.csv
    Contains all combined results per object.
    Results_$Mask_Filename_$Intensity_Image_filename.csv
    Contains results per object for the defined pair of images.
    Combined_Object_Data_Analysis_ID_$Data_Analysis_ID_$Search_Term_you_used_to_filter_results.csv
    Contains all post-processed results per image (e.g. cell number, mean intensity, ect.)
    '''
    stats_per_folder = []
    count = 0
    for key,value in mask_to_img_dic.items():
        stats_per_img = {}
    # load images
        ma = tif.imread(key)
        im = tif.imread(value)
    # get results per object
        results = skimage.measure.regionprops_table(ma, intensity_image=im, properties=get_property, cache=True)
        results_df = pd.DataFrame.from_records(results)
        results_df["Mask_Image"] = os.path.split(key)[1]
        results_df["Intensity_Image"] = os.path.split(value)[1]
        results_df["sum_intensity"] = results_df["mean_intensity"] * results_df["area"]
        results_df["circularity"] = results_df["equivalent_diameter"] * math.pi / results_df["perimeter"]
        new_order = ["Mask_Image","label","area","centroid-0","centroid-1"] + get_property[4:] + ["sum_intensity","circularity","Intensity_Image"]
        results_df = results_df.reindex(columns = new_order)
        # to avoid confusion for Fiji user
        results_df.rename(columns={'centroid-0':'centroid-0_Fiji_Y'}, inplace=True)
        results_df.rename(columns={'centroid-1':'centroid-1_Fiji_X'}, inplace=True)
        new_fn = "Results_{}_{}.csv".format(os.path.split(key)[1],os.path.split(value)[1])
        new_fp = os.path.join(root,"Processed_{}".format(run_ID),sub_f[output_folder],new_fn)
        results_df.to_csv(new_fp, sep=';', decimal=',')
    # get results per image
        stats_per_img["Mask"] = os.path.split(key)[1]
        stats_per_img["Intensity_Image"] = os.path.split(value)[1]
        stats_per_img["count"] = results_df.shape[0]
        stats_per_img["median_area"] = results_df["area"].median()
        stats_per_img["mean_area"] = results_df["area"].mean()
        stats_per_img["mean_intensity"] = results_df["mean_intensity"].mean()
        stats_per_img["median_circularity"] = results_df["circularity"].median()
        stats_per_img["median_sum_intensity"] = results_df["sum_intensity"].median()
        stats_per_folder.append(stats_per_img)
        if count > 0:
            all_data = pd.concat([all_data,results_df])
        else:
            all_data = results_df
        count += 1
    # save combined object data
    new_fn = "Combined_Object_Data_{}_{}.csv".format("_".join(subset),tag)
    new_fp = os.path.join(root,"Processed_{}".format(run_ID),sub_f[output_folder],new_fn)
    all_data.to_csv(new_fp, sep=';', decimal=',')
    # save summary data
    all_results_df = pd.DataFrame.from_records(stats_per_folder)
    new_fn = "Summary_Results_{}_{}.csv".format("_".join(subset),tag)
    new_fp = os.path.join(root,"Processed_{}".format(run_ID),sub_f[output_folder],new_fn)
    all_results_df.to_csv(new_fp, sep=';', decimal=',')
    return all_data

def create_mask_from_add_ch(exfl,root,sub_f,run_ID,para_mask_post):
    '''
    function to create masks from the additional channel,
    only called if create_filter_mask_from_channel == True
    is just one way to post-process & filter results.
    Alternatively any software might be used to create files that follow these naming scheme:
    Next, these masks might be used to analyse only parts of the image.
    Main_Folder/AdditionalMask/
    Mask_$FN
    binary mask used to filter results in the post_processing'''
    for img_f in exfl:
        p,f = os.path.split(img_f)
        fnew = f.replace("Sum_","Mask_")
        img_name = os.path.join(root,"Processed_{}".format(run_ID),sub_f[6],fnew)
        myimg = tif.imread(img_f)
        #print(myimg.shape)
        proj_img_8bit = (255 * myimg / np.max(myimg)).astype(np.uint8)
        bin_img = get_binary(proj_img_8bit,para_mask_post)
        tif.imsave(img_name,bin_img)
    new_mask_fn_list = glob("{}/*tif".format(os.path.join(root,"Processed_{}".format(run_ID),sub_f[6])))
    return new_mask_fn_list

# naming scheme DemoOpSef 4
def make_pair_dic_simple(masks_input_files,MD,CQ,root,run_ID,sub_f,which_subfolder = 0):
    # makes dic of filepath for mask to base image that will be used for quantification
    mask_to_img = {}
    for mask_fn in masks_input_files:
        split_fn = os.path.split(mask_fn)[1].split("_")
        new_fn = "_".join([CQ[0]]+split_fn[MD[0]:])
        fp_new = os.path.join(root,"Processed_{}".format(run_ID),sub_f[which_subfolder],new_fn)
        mask_to_img[mask_fn] = fp_new
    return mask_to_img

# naming scheme DemoOpSef 4
def make_pair_second_mask_simple(masks_input_files,new_mask_fn_list):
    # makes dic of filepath for orginal (segmentation-generated)
    # mask to the new mask image generated by an additional channel
    mask_to_img = {}
    for mask_fn in masks_input_files:
        split_fn = os.path.split(mask_fn)[1].split("_")
        root_fn = "_".join(split_fn[5:])
        match = [x for x in new_mask_fn_list if root_fn in x][0]
        mask_to_img[mask_fn] = match
    return mask_to_img

def combine_masks(label_img,thresholded_image):
    # this function uses a mask to split
    # the cells in two classes
    # class 2 is inside the provided mask
    # class 1 outside the
    cells_class1 = label_img.copy()
    cells_class2 = label_img.copy()
    # get region properties
    props = regionprops(label_img)
    regions_class1 = []
    regions_class2 = []
    # filter regions
    for region in props:
        if region.label > 0:
            x,y = region.centroid
            if thresholded_image[int(x),int(y)] == 0:
                regions_class1.append(region.label)
            else:
                regions_class2.append(region.label)
    # define class 1
    for mv in regions_class1:
        if mv > 0:
            mask_this_cell = np.zeros(label_img.shape,dtype=bool)
            mtc = np.where(label_img == mv)
            mask_this_cell[mtc] = 1
            cells_class1[mask_this_cell] = 0 # delete this cell
    for mv in regions_class2:
        if mv > 0:
            mask_this_cell = np.zeros(label_img.shape,dtype=bool)
            mtc = np.where(label_img == mv)
            mask_this_cell[mtc] = 1
            cells_class2[mask_this_cell] = 0 # delete this cell
    return cells_class1,cells_class2

def split_by_mask(root,run_ID,sub_f,pair_dic,mask_to_8bitimg_dic,mask_to_img_dic):
    '''
    Splits the results of the original segementation in two classes
    Class 1 are cells that have a center of mass inside the provided threshold mask,
    Class 2 are cells that have a center of mass outside the provided threshold mask.
    Three results will be saved:
    MaskClass01
    MaskClass02
    and an overlay image in which Class1 objects are displayed in red, Class 2 objects in green
    returns these two dictonaries:
    class1_dic[class1_segmentation_mask_name] = intensity_image_to_be_quantified
    class2_dic[class2_segmentation_mask_name] = intensity_image_to_be_quantified'''
    # create filename lists:
    class1_dic = {}
    class2_dic = {}
    for key,value in pair_dic.items():
    # load images
        seg_mask = tif.imread(key)
        bin_mask = tif.imread(value)
        overlay_img_name = mask_to_8bitimg_dic[key]
        overlay_img = tif.imread(overlay_img_name)
    # create combined masks
        img_class1,img_class2 = combine_masks(seg_mask,bin_mask)
    # save new masks
        p,f = os.path.split(key)
        fnew1 = f.replace("_Mask_","_MaskClass01_")
        img_name1 = os.path.join(root,"Processed_{}".format(run_ID),sub_f[7],fnew1)
        tif.imsave(img_name1,img_class1)
        class1_dic[img_name1] = mask_to_img_dic[key]
        fnew2 = f.replace("_Mask_","_MaskClass02_")
        img_name2 = os.path.join(root,"Processed_{}".format(run_ID),sub_f[7],fnew2)
        tif.imsave(img_name2,img_class2)
        class2_dic[img_name2] = mask_to_img_dic[key]
    # create overlay
        overlay = draw_2outlines(overlay_img,[img_class1,img_class2])
        fnew3 = f.replace("_Mask_","_Overlay_by_Class_")
        img_name3 = os.path.join(root,"Processed_{}".format(run_ID),sub_f[8],fnew3)
        tif.imsave(img_name3,overlay)
    return class1_dic,class2_dic

def extract_values_for_TSNE_PCA(root,run_ID,sub_f,which_subfolder,include_in_tsne):
    '''Extracts the values needed for TSNE and PCA analysis
    as defined in include_in_tsne.
    A dataframe will be returned'''
    for_tsne = []
    results_file_name = glob(os.path.join(root,"Processed_{}".format(run_ID),sub_f[which_subfolder])+"/*Combined_Object*.csv")
    for rf in results_file_name:
        df_read = pd.read_csv(rf,sep=";",decimal=",")
        df_extract = df_read.filter(include_in_tsne,axis=1)
        for_tsne.append(df_extract)
    return for_tsne

def draw_2outlines(img,mask_list):
    # shows outline of a mask in red
    RGB = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    for zz in (0,1,2):
        RGB[:,:,zz] = img[:,:].copy()
    for i,mask in enumerate(mask_list):
        mask_bin = np.zeros(mask.shape, dtype = bool)
        mask_bin[np.where(mask > 0)] = 1
        mask_bin = cv_binary_processor_plus(mask_bin,[["dilate",3,1,1,"Morphology"]]) # erode
        core = get_core_img(mask)
        mask_bin[core] = 0 # delete center
        paint = np.zeros(mask.shape, dtype = np.uint8)
        paint[:,:] = RGB[:,:,i]
        paint[mask_bin] = 255
        RGB[:,:,i] = paint[:,:]
        if i == 0:
            RGB[:,:,2] = paint[:,:] # blue channel = red channel
    return RGB

def merge_intensity_results(all_combined,input_def,sub_f,run_def,output_folder = 4):
    '''merges the results created by applying the ame mask on two intesity channel
    as e.g. a DAPI  EdU pipeline'''
    result_summary = all_combined[0].copy()
    result_summary['mean_int_2nd_ch'] = all_combined[1]["mean_intensity"].copy()
    result_summary['sum_int_2nd_ch'] = all_combined[1]["sum_intensity"].copy()
    result_summary['ratio'] = result_summary['mean_int_2nd_ch'] / result_summary["mean_intensity"]
    new_fn = "Combined_Object_Data_{}_{}.csv".format("_".join(input_def["subset"]),"All_Channel")
    new_fp = os.path.join(input_def["root"],"Processed_{}".format(run_def["run_ID"]),sub_f[output_folder],new_fn)
    result_summary.to_csv(new_fp, sep=';', decimal=',')
    return result_summary

def segment(input_def, pc, run_def,initModelSettings):
    # runs the segmentation
    # define input:
    input_path = os.path.join(input_def["root"], "Processed_{}".format(run_def["run_ID"]), pc["sub_f"][1])
    files = glob("{}/*RGBInput*.tif".format(input_path))
    file_name_core = [os.path.split(x)[1] for x in files]
    results_dic_list = []
    # load model and process data
    for run_sub_id,run_now in enumerate(run_def["run_now_list"]):
        print("run_now", run_now)
        im_id = 0
        # load model Cellpose
        if run_now == "Cellpose":
            print("Run Cellpose Model: ",run_def["ModelType"][run_sub_id])
            if run_def["ModelType"][run_sub_id] == "CP_nuclei":
                my_model = initialize_model(initModelSettings,run_now,"CP_nuclei")
            elif run_def["ModelType"][run_sub_id] == "CP_cyto":
                my_model = initialize_model(initModelSettings,run_now,"CP_cyto")
        # load model StarDist
        elif run_now == "StarDist":
            print("Run StarDist",run_def["ModelType"][run_sub_id])
            if run_def["ModelType"][run_sub_id] == "SD_2D_dsb2018":
                print("Run_SD_",run_now)
                my_model = initialize_model(initModelSettings,run_now,"SD_2D_dsb2018")
        elif run_now == "Cellprofiler_UNet":
            if run_def["ModelType"][run_sub_id] == "UNet_CP001":
                print("Run Cellprofiler_UNet",run_def["ModelType"][run_sub_id])
                my_model = initialize_model(initModelSettings,run_now,"UNet_CP001")
        # run model on data
        while (im_id + pc["batch_size"]) <= len(files):
            # load data for batch
            batch = [plt.imread(f) for i, f in enumerate(files) if i in range(im_id, (im_id + pc["batch_size"]))]
            batch_fn = [os.path.split(x)[1] for i, x in enumerate(files) if
                        i in range(im_id, (im_id + pc["batch_size"]))]
            # sometimes it makes sense to use very blurred images as input to the classifier
            # if these shall not be used to display the results an alternative input may be defined
            # the processing parameter for this input is called "display base"
            if run_def["display_base"] != "same":
                base_fn_in = ["_".join([x.split("_")[0]] + [run_def["display_base"]] + x.split("_")[2:]) for x in
                              batch_fn]
                fp_in = [os.path.join(input_def["root"], "Processed_{}".format(run_def["run_ID"]), pc["sub_f"][1], x) for x in
                         base_fn_in]
                batch_display = [plt.imread(x) for x in fp_in]
            else:
                batch_display = []
            print("Running now: ", batch_fn)
            if run_now == "Cellpose":
                results_dic_list.append(
                    run_batch_cellpose(run_sub_id,my_model,run_def["ModelType"][run_sub_id], input_def["root"], batch, batch_display, batch_fn,
                                       run_def["rescale_list"], initModelSettings["Cell_Channels"],
                                       run_def["filter_para"], pc["sub_f"], run_def["run_ID"]))
            elif run_now == "StarDist":
                results_dic_list.append(run_batch_stardist(run_sub_id,my_model,run_def["ModelType"][run_sub_id], input_def["root"], batch, batch_display, batch_fn,
                                                           run_def["filter_para"], pc["sub_f"], run_def["run_ID"]))
            elif run_now == "Cellprofiler_UNet":
                results_dic_list.append(
                    run_batch_Unet(run_sub_id,my_model,run_def["ModelType"][run_sub_id], input_def["root"], batch, batch_display, batch_fn, run_def["filter_para"],
                                   pc["sub_f"], run_def["run_ID"]))
            im_id += 2
            print(im_id)
    return results_dic_list