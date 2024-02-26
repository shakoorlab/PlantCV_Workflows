#!/usr/bin/env python

import sys, traceback
import cv2
import os
import re
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv


def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-r2","--coresult", help="result file.", default=None )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", action="store_true")
    args = parser.parse_args()
    return args

### Main pipeline
def main():
    # Get options
    args = options()
    # Parameterize starting coord and spacing of color card. 
    start = (1312, 1892)
    space = (47,47)
    
    # Read image 
    img, path, filename = pcv.readimage(args.image)
    
    # Source image matrix from mask 
    colorcard_mask = pcv.transform.create_color_card_mask(rgb_img=img, radius=10, start_coord=start, 
                                                   spacing=space, nrows=4, ncols=6)
    headers, s_matrix = pcv.transform.get_color_matrix(rgb_img=img, mask=colorcard_mask)
    
    # Load in x-rite color card matrix
    t_matrix = pcv.transform.load_matrix(filename="/shares/bioinformatics/hschuhl/projects/fonio/x-rite_color_matrix.npz")
    # get matrix_m
    matrix_a, matrix_m, matrix_b = pcv.transform.get_matrix_m(target_matrix=t_matrix, source_matrix=s_matrix)
    # calc tranformation matrix 
    deviance, transformation_matrix = pcv.transform.calc_transformation_matrix(matrix_m, matrix_b)
    
    # apply transformation, use source image twice since we dont have target color card img
    # and it doesn't matter since the target img is just used for plotting 
    corrected_img_x = pcv.transform.apply_transformation_matrix(img, img, transformation_matrix)
    
    # Segmentation 
    l = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    l_thresh = pcv.threshold.binary(gray_img=l, threshold=167, max_value=255, object_type='dark')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=70, max_value=255, object_type='light')
    combined_mask = pcv.logical_or(bin_img1=l_thresh, bin_img2=s_thresh)
    closed_mask = pcv.closing(gray_img=combined_mask)
    filled_mask = pcv.fill(bin_img=closed_mask, size=45)
    blurred_mask = pcv.median_blur(gray_img=filled_mask, ksize=2)
    
    # ROI and filter 
    obj_c, obj_hierarchy = pcv.find_objects(img=img, mask=blurred_mask)
    roi_c, roi_hierarchy= pcv.roi.custom(img=img, vertices=[[600,200], [1800,200], [1800, 1400], [1450,1400], 
                                                        [1450,1056], [1030, 1056], [1030,1400], [600,1400]])
    plant_objects, plant_hierarchy, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi_c, 
                                                               roi_hierarchy=roi_hierarchy, 
                                                               object_contour=obj_c, 
                                                               obj_hierarchy=obj_hierarchy,
                                                               roi_type='cutto')
    obj, plant_mask = pcv.object_composition(img=img, contours=plant_objects, hierarchy=plant_hierarchy)
    
    # Analysis 
    color_histogram = pcv.analyze_color(rgb_img=corrected_img_x, mask=plant_mask, colorspaces='all', label="default")
    analysis_image = pcv.analyze_object(img=img, obj=obj, mask=plant_mask, label="default")
    boundary_image = pcv.analyze_bound_horizontal(img=img, obj=obj, mask=plant_mask, 
                                               line_position=1070, label="default")
    
    # Output shape and color data + debug img 
    pcv.print_image(analysis_image,os.path.join(args.outdir,filename +'_shape.jpg'))
    pcv.outputs.save_results(args.result)  
    
if __name__ == '__main__':
    main()

