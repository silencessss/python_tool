from PIL import Image, ImageOps
import collections
import json
import os
import cv2
import math
import argparse
import sys
import time
import imutils
import numpy as np
####################################################################################################

####################################################################################################
def merge_image(image,x,y,stepW,stepH,nums,image_name,bg_path):
    #basic setting
    img_froeground=image
    background_path=bg_path
    img_background=cv2.imread(background_path)
    x_start=x
    y_start=y
    x_end=x+stepW
    y_end=y+stepH
    #merge..
    img_background[y_start:y_end,x_start:x_end]=img_froeground
    #save
    count=nums
    if count<=4:
        save_path='./merge_image_original_bk_done/'+image_name+'_cut_'+str(count)+'.png'
        cv2.imwrite(save_path,img_background)
    #retutn
    return img_background
####################################################################################################
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image
####################################################################################################
def sliding_window(x_start,y_start,x_end,y_end,image, set_stepSize, windowSize):
    x_start_sliding_window=x_start
    y_start_sliding_window=y_start
    x_end_sliding_window=x_end
    y_end_sliding_window=y_end
    (set_stepSize_x,set_stepSize_y)=set_stepSize
    # slide a window across the image
    for y in range(y_start_sliding_window, y_end_sliding_window, set_stepSize_y):
        for x in range(x_start_sliding_window, x_end_sliding_window, set_stepSize_x):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
####################################################################################################
def bndbox(points):
        """
        generate bounding box from anchors
        parameter(s)
            points      list    coordinate of all points in (x, y)
        return
            bbox        tuple   (xmin, ymin, xmax, ymax)
        """
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        return min(x), min(y), max(x), max(y)
####################################################################################################
def main_sliding(image,start_point,sliding_window_size,img_name,background_path,jsn,set_stepSize):
    (x_start,y_start,x_end,y_end,set_stepSize)=start_point
    (W_sliding_window,H_sliding_window)=sliding_window_size
    count=1
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(x_start,y_start,x_end,y_end,resized, set_stepSize, windowSize=(W_sliding_window, H_sliding_window)):
            x4=x+W_sliding_window
            y4=y+H_sliding_window
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != H_sliding_window or window.shape[1] != W_sliding_window:
                continue
            
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x-1, y-1), (x + W_sliding_window, y + H_sliding_window), (0, 255, 0), 1)
            img_sliding_crop=clone[y:y + H_sliding_window,x:x + W_sliding_window]
            img_merge=merge_image(img_sliding_crop,x,y,W_sliding_window,H_sliding_window,count,img_name,background_path)
            
            cv2.imshow('crop image',img_sliding_crop)
            cv2.imshow('merge image',img_merge)
            if count<=4:
                print('(x1,y1)       ',(x,y))
                print('(x4,y4)       ',(x+W_sliding_window,y+H_sliding_window))
                print('next...')
                #save_path='./subimage_600x600_nospace_singleobject/'+img_name+'_'+str(count)+'.png'
                #merge_image_save_path='./merge_image_600x600_nospace_singleobject/'+img_name+'_'+str(count)+'.png'
                merge_image_save_name=img_name+'_'+str(count)+'.png'
                #cv2.imwrite(save_path,img_sliding_crop)
                
                jsn['imageWidth']  = img_merge.shape[0]
                jsn['imageHeight'] = img_merge.shape[1]
                jsn['imagePath']   = merge_image_save_name
                new_jsn_basename = img_name + '_'+str(count)+'.json'
                new_jsn_save_dir= './merge_image_original_bk_done/'
                new_jsn_filepath = os.path.join(new_jsn_save_dir, new_jsn_basename)
                '''
                for shape in jsn['shapes']:
                    #labels.append(shape['label'])
                    points = shape['points']
                    shape['points'] = [
                        [x, y],
                        [x4, y4]
                        ]
                    shape['shape_type'] = 'rectangle'
                '''
                with open(new_jsn_filepath, 'w') as fp:
                    json.dump(jsn, fp, indent=4)       
                
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.8000)
            count+=1
####################################################################################################
def basic_setting(filepath):
    img_filepath = filepath
    print('img_filepath ==',img_filepath)
    # background
    background_path='background03.png'
    img_basename = os.path.basename(img_filepath)#include(.jpg)
    print('img_basename ==',img_basename)
    image=cv2.imread(img_filepath)

    # get image name which no file format
    img_name, img_ext = os.path.splitext(img_basename)

    # read *.json files and get bounding box point
    jsn_filepath = img_filepath.replace(img_ext, '.json')
    with open(jsn_filepath, 'r') as fp:
        jsn = json.load(fp)
    
    bndboxes = list()
    labels   = list()
    for shape in jsn['shapes']:
        labels.append(shape['label'])
        points = shape['points']
        for idx, point in enumerate(points):
            x, y = point
            points[idx] = [x, y]
        xmin, ymin, xmax, ymax = bndbox(shape['points'])
    xmin=round(xmin)
    ymin=round(ymin)
    xmax=round(xmax)
    ymax=round(ymax)

    #get bounding box size(W_BB,H_BB)
    W_BB = abs(xmax-xmin)
    H_BB = abs(ymax-ymin)

    #setting the sliding window size(W_slidiing_window,H_sliding_window)(0.8)
    proportion=0.8
    W_sliding_window = round(W_BB*proportion)
    H_sliding_window = round(H_BB*proportion)
    sliding_window_size=(W_sliding_window,H_sliding_window)

    # stepsize and start point(xmin,ymin)
    # stepsize
    set_stepSize_x=round(W_BB*(1-proportion))
    set_stepSize_y=round(H_BB*(1-proportion))
    set_stepSize=(set_stepSize_x,set_stepSize_y)
    # start point(x_start,y_start,x_end,y_end)
    x_start=round(xmin)
    y_start=round(ymin)
    x_end=round(xmin)+round(W_BB*(1-proportion))*2
    y_end=round(ymin)+round(H_BB*(1-proportion))*2
    start_point=(x_start,y_start,x_end,y_end,set_stepSize)
    
    # main_sliding()
    main_sliding(image,start_point,sliding_window_size,img_name,background_path,jsn,set_stepSize)
    # bounding box area and crop the image
    image_bounding_box_area = image[ymin:ymin+H_BB, xmin:xmin+W_BB]
    cv2.imwrite(img_name+'.png',image_bounding_box_area)
    #cv2.imshow('window',image_bounding_box_area)
    #print(int(image_bounding_box_area.shape[0]))
####################################################################################################
if __name__=='__main__':
    #walk all the files
    parser = argparse.ArgumentParser(description='Resize images in a folder')
    parser.add_argument('--input',    '-i', help='input image folder', required=True)
    args = parser.parse_args()
    input_path = os.path.abspath(args.input)
    if not os.path.isdir(args.input):
        print('--input ({}) must be a folder.'.format(args.input))
        sys.exit(1)
    count=1
    for root, _, basenames in os.walk(input_path):
        for basename in basenames:
            basename_check=basename.split('.')[1]
            if(basename_check=='jpg' or basename_check=='png'):
                print('processing...',count)
                filepath = os.path.join(root, basename)
                basic_setting(filepath)
                count+=1


    
    