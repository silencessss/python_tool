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
            #print('basename == ',basename)
            basename_check=basename.split('.')[1]
            if(basename_check=='jpg'):
                print('processing...',count)
                print('image is ....',basename)
                filepath = os.path.join(root, basename)
                with open('get_image_name_tmp.txt','a+') as fw:
                    fw.write(basename.split('.')[0]+'\n')
                    print('write success')
                count+=1