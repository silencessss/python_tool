import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import argparse
import json

def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
	# get half size
	d = K_size // 2
	# prepare kernel
	gabor = np.zeros((K_size, K_size), dtype=np.float32)
	# each value
	for y in range(K_size):
		for x in range(K_size):
			# distance from center
			px = x - d
			py = y - d
			# degree -> radian
			theta = angle / 180. * np.pi
			# get kernel x
			_x = np.cos(theta) * px + np.sin(theta) * py
			# get kernel y
			_y = -np.sin(theta) * px + np.cos(theta) * py
			# fill kernel(the part of Real Number)
			gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

	# kernel normalization
	gabor /= np.sum(np.abs(gabor))

	return gabor


def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape
    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')
    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)
    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def Gabor_process(img):
    # get shape
    H, W = img.shape
    # gray scale
    #gray = BGR2GRAY(img).astype(np.float32)
    gray=img
    # define angle = [0, 45, 90, 135]
    As = [0,30,60,90,120,150]
    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)
        #cv2.imwrite('out_angle_'+str(A)+'.jpg',_out)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out

if __name__=='__main__':
    # walk all the files
    parser = argparse.ArgumentParser(description='Filter images in a folder')
    parser.add_argument('--input','-i',default='./datasets/Data_original/image_original_nospace_newlabel/', help='input image folder')
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
                # base path
                filepath = os.path.join(root, basename)
                jsn_filepath = filepath.replace(basename_check, 'json')
                print(filepath)
                print(jsn_filepath)
                # Read image
                img = cv2.imread(filepath).astype(np.float32)
                (img_B,img_G,img_R)=cv2.split(img)
                # gabor process(gray || rgb)
                out_B = Gabor_process(img_B)
                out_G = Gabor_process(img_G)
                out_R = Gabor_process(img_R)
                out_merge = cv2.merge([out_B,out_G,out_R])
                # new path
                new_filepath='./filter_out/main/'+str(basename.split('.')[0])
                new_img_filepath = new_filepath+'_gabor.jpg'
                new_jsn_filepath = new_filepath+'_gabor.json'
                print(new_img_filepath)
                print(new_jsn_filepath)
                # save
                cv2.imwrite("./filter_out/B/"+save_name_out+"_B.jpg", img_B)
                cv2.imwrite("./filter_out/G/"+save_name_out+"_G.jpg", img_G)
                cv2.imwrite("./filter_out/R/"+save_name_out+"_R.jpg", img_R)
                cv2.imwrite(new_img_filepath, out_merge)
                # new json
                with open(jsn_filepath, 'r') as fp:
                    jsn = json.load(fp)
                #jsn['imageWidth']  = out_merge.shape[0]
                #jsn['imageHeight'] = out_merge.shape[1]
                jsn['imagePath']   = str(basename.split('.')[0])+"_gabor.jpg"
                with open(new_jsn_filepath, 'w') as fp:
                    json.dump(jsn, fp, indent=4)
                count+=1