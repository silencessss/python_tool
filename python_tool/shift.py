import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import argparse
import json

def shift(img,x,y):
    '''
    official:   cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
    custom  :   cv2.warpAffine(img,shift_array,(shifted_rows,shifted_cols))
    '''
    Arr_shift=np.float32([[1, 0, x], [0, 1, y]])
    shifted=cv2.warpAffine(img, Arr_shift, (img.shape[1], img.shape[0]))
    return shifted

def base_set(path, basename, direction):
    img=cv2.imread(path)
    H,W,_=img.shape
    '''
    Here to set the shifted percent!!!
    direction = 0 - shift left, 1 - shift right, 2 - shift down, 3 - shift up
    '''
    if direction == 0:
        x_percent=0.1
        y_percent=0.0
        x_shift = -round(H*x_percent)   #set '+' or '-'
        y_shift = round(W*y_percent)    #set '+' or '-'
    elif direction == 1:
        x_percent=0.1
        y_percent=0.0
        x_shift = round(H*x_percent)    #set '+' or '-'
        y_shift = round(W*y_percent)    #set '+' or '-'
    elif direction == 2:
        x_percent=0.0
        y_percent=0.1
        x_shift = round(H*x_percent)    #set '+' or '-'
        y_shift = round(W*y_percent)    #set '+' or '-'
    else:
        x_percent=0.0
        y_percent=0.1
        x_shift = round(H*x_percent)    #set '+' or '-'
        y_shift = -round(W*y_percent)   #set '+' or '-'

    shifted=shift(img,x_shift,y_shift)

    jsn_filepath = path.replace(basename.split('.')[1],'json')
    new_jsn_filepath = './out_shifted'+str(direction)+'/'+str(basename.split('.')[0])+'_shifted'+str(direction)+'.json'
    print(jsn_filepath)

    with open(jsn_filepath, 'r') as fp:
        jsn = json.load(fp)
    for shape in jsn['shapes']:
        points = shape['points']
        for idx, point in enumerate(points):
            x, y = point
            x = x+x_shift
            y = y+y_shift
            points[idx] = [x, y]
    jsn['imagePath']   = str(basename.split('.')[0])+"_shifted"+str(direction)+".jpg"
    with open(new_jsn_filepath, 'w') as fp:
        json.dump(jsn, fp, indent=4)

    return shifted

if __name__=='__main__':
    # walk all the files
    parser = argparse.ArgumentParser(description='Filter images in a folder')
    parser.add_argument('--input','-i',default='./datasets/smartRetail/evalImage_640x640_noSpace/', help='input image folder')
    args = parser.parse_args()
    input_path = os.path.abspath(args.input)
    if not os.path.isdir(args.input):
        print('--input ({}) must be a folder.'.format(args.input))
        sys.exit(1)
    count=1
    for root, _, basenames in os.walk(input_path):
        for basename in basenames:
            basename_check=basename.split('.')[1]
            if(basename_check=='jpg'):
                print('processing...',count)
                count+=1
                img_filepath = os.path.join(root,basename)
                print(img_filepath)
                direction = 0
                while (direction < 4):
                    # shifted function
                    shifted=base_set(img_filepath, basename, direction)
                    # save
                    save_filename_path='./out_shifted'+str(direction)+'/'+str(basename.split('.')[0])+'_shifted'+str(direction)+'.jpg'
                    cv2.imwrite(save_filename_path,shifted)
                    direction +=1

                print('4 image saved!!!')