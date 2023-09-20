
from plantcv import plantcv as pcv
import matplotlib
import cv2
import numpy as np
import argparse
from  matplotlib import pyplot as plt
import os
from plantcv.parallel import workflow_inputs
import datetime

start = datetime.datetime.now()

# Set input variables
args = workflow_inputs() 

print(args.image1)    
# Set variables
pcv.params.debug = args.debug     # Replace the hard-coded debug with the debug flag

#use image1 because of the new workflow inputs
img, path, filename = pcv.readimage(filename=args.image1)
filename = os.path.split(args.image1)[1]

#define new color correct function for logv images
def affine_color_correction(img, source_matrix, target_matrix):
    h,w,c = img.shape
    n = source_matrix.shape[0]
    S = np.concatenate((source_matrix[:,1:].copy(),np.ones((n,1))),axis=1)
    T = target_matrix[:,1:].copy()
    
    tr = T[:,0]
    tg = T[:,1]
    tb = T[:,2]
    
    ar = np.matmul(np.linalg.pinv(S), tr)
    ag = np.matmul(np.linalg.pinv(S), tg)
    ab = np.matmul(np.linalg.pinv(S), tb)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pix = np.concatenate((img_rgb.reshape(h*w,c).astype(np.float64)/255, np.ones((h*w,1))), axis=1)
    
    img_r_cc = (255*np.clip(np.matmul(img_pix,ar),0,1)).astype(np.uint8)
    img_g_cc = (255*np.clip(np.matmul(img_pix,ag),0,1)).astype(np.uint8)
    img_b_cc = (255*np.clip(np.matmul(img_pix,ab),0,1)).astype(np.uint8)
    
    img_cc = np.stack((img_b_cc,img_g_cc,img_r_cc), axis=1).reshape(h,w,c)
    
    return img_cc

#log correct the plant image  #for 2023 cut out everything to right of '=' and replace it with 'img' so plact_logv = img 
plant_logv = img  


#define the color card dataframe for the image you are playing with
dataframe2, start2, space2 = pcv.transform.find_color_card(rgb_img=plant_logv, background='light')

#make a color card mask for your image that you are playing with here

source_mask = pcv.transform.create_color_card_mask(plant_logv, radius=10, start_coord=start2, 
                                                   spacing=space2, nrows=4, ncols=6)

#make color card matrix for image you are playing with
headers, source_matrix = pcv.transform.get_color_matrix(rgb_img=plant_logv, mask=source_mask)

#load the color card values that they should be
target_matrix = pcv.transform.load_matrix(filename='/shares/nshakoor_share/users/jstanton/phenotyper_data/2022/x-rite_color_matrix_k2.npz')

color_corrected_img = affine_color_correction(plant_logv, source_matrix, target_matrix)


#box on right 
box_right_img, binary, contours, hierarchy = pcv.rectangle_mask(img=color_corrected_img, p1=(3050,2696), p2=(3300,2), color = "white")

#creating box on left 
box_left_and_right_img, binary, contours, hierarchy = pcv.rectangle_mask(img=box_right_img, p1=(770,2688), p2=(912,2), color = "white")

thresh1 = pcv.threshold.dual_channels(rgb_img = box_left_and_right_img, x_channel = "a", y_channel = "b", points = [(105,130),(130,140)], above=True, max_value=255)


#get rid of noise
thresh1_fill = pcv.fill(bin_img=thresh1, size=2000)


# Fill in small objects #does not even take a sizing parameter #obviouspepper
thresh1_filled_holes = pcv.closing(gray_img=thresh1_fill)


id_objects_ab, obj_hierarchy_ab = pcv.find_objects(img=color_corrected_img, mask=thresh1_filled_holes)


roi_ab, roi_hierarchy_ab= pcv.roi.rectangle(img=color_corrected_img, x=1250, y=700, h=1275, w=1700)


roi_objects_ab, hierarchy_ab, kept_mask_ab, obj_area_ab = pcv.roi_objects(img=color_corrected_img, roi_contour=roi_ab, 
                                                               roi_hierarchy=roi_hierarchy_ab, 
                                                               object_contour=id_objects_ab, 
                                                               obj_hierarchy=obj_hierarchy_ab,
                                                               roi_type='partial')

if obj_area_ab > 4:
    #combine kept objects
    obj_combined_ab, kept_mask_ab = pcv.object_composition(img=color_corrected_img, contours=roi_objects_ab, hierarchy=hierarchy_ab)

    # Find shape properties, data gets stored to an Outputs class automatically
    analysis_image = pcv.analyze_object(img=color_corrected_img, obj=obj_combined_ab, mask=kept_mask_ab, label="default")
    boundary_image = pcv.analyze_bound_horizontal(img=color_corrected_img, obj=obj_combined_ab, mask=kept_mask_ab, 
                                               line_position=1980, label="default")

# Determine color properties
    color_histogram = pcv.analyze_color(rgb_img=color_corrected_img, mask=kept_mask_ab, colorspaces='all', label="default")

    pcv.print_image(analysis_image, os.path.join(args.outdir, filename + "_result.jpg"))

    pcv.outputs.save_results(filename=args.result)



end = datetime.datetime.now()
duration = end - start
print(f"{args.image1}: {duration.seconds}")





