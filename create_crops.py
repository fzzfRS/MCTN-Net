#!/usr/bin/env python2

"""
create_crops.py: script to crops for training and validation.
                It will save the crop images and mask in the format: 
                <image_name>_<x>_<y>.<suffix>
                where x = [0, (Image_Height - crop_size) / stride]
                      y = [0, (Image_Width - crop_size) / stride] 

It will create following directory structure:
    base_dir
        |   train_crops.txt   # created by script
        |   val_crops.txt     # created by script
        |
        └───train_crops       # created by script
        │   └───gt
        │   └───images
        └───val_crops         # created by script
        │   └───gt
        │   └───images
        └───test_crops        # created by script
        │   └───gt
        │   └───images

"""

from __future__ import print_function

import argparse
import os
import mmap
import cv2
import time
import numpy as np
from skimage import io
from tqdm import tqdm
tqdm.monitor_interval = 0



def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def CreatCrops(base_dir, crop_type, size, stride, image_suffix, gt_suffix):

    crops = os.path.join(base_dir, '{}_crops'.format(crop_type))
    crops_file = open(os.path.join(base_dir,'{}_crops.txt'.format(crop_type)),'w')

    full_file_path = os.path.join(base_dir,'{}.txt'.format(crop_type))
    full_file = open(full_file_path,'r')

    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    failure_images = []
    for name in tqdm(full_file, ncols=100, desc="{}_crops".format(crop_type), 
                            total=get_num_lines(full_file_path)):
        
        name = name.strip("\n")
        image_file = os.path.join(base_dir,'{}/images'.format(crop_type),name+image_suffix)
        gt_file = os.path.join(base_dir,'{}/gt'.format(crop_type),name+gt_suffix)

        if not verify_image(image_file):
            failure_images.append(image_file)
            continue

        image = cv2.imread(image_file)
        gt = cv2.imread(gt_file,0)
        
        if image is None:
            failure_images.append(image_file)
            continue

        if gt is None:
            failure_images.append(image_file)
            continue

        H,W,C = image.shape
        maxx = (H-size)//stride
        maxy = (W-size)//stride
        
        for x in range(maxx+1):
            for y in range(maxy+1):
                im_ = image[x*stride:x*stride + size,y*stride:y*stride + size,:]
                gt_ = gt[x*stride:x*stride + size,y*stride:y*stride + size]
                crops_file.write('{}_{}_{}\n'.format(name,x,y))
                cv2.imwrite(crops+'/images/{}_{}_{}.tif'.format(name,x,y),  im_)
                cv2.imwrite(crops+'/gt/{}_{}_{}.tif'.format(name,x,y), gt_)
    
    crops_file.close()
    full_file.close()
    if len(failure_images) > 0:
        print("Unable to process {} images : {}".format(len(failure_images), failure_images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, default=r'./data/MCTNDataset/',
        help='Base directory for Deepglobe Dataset.')
    parser.add_argument('--crop_size', type=int, default=512,
        help='Crop Size of Image')
    parser.add_argument('--crop_overlap', type=int, default=256,
        help='Crop overlap Size of Image')
    parser.add_argument('--im_suffix', type=str, default='.tif',
        help='Dataset specific image suffix.')
    parser.add_argument('--gt_suffix', type=str, default='.tif',
        help='Dataset specific gt suffix.')

    args = parser.parse_args()

    start = time.clock()
    ## Create overlapping Crops for training
    CreatCrops(args.base_dir,
                crop_type='train',
                size=args.crop_size,
                stride=args.crop_overlap,
                image_suffix=args.im_suffix,
                gt_suffix=args.gt_suffix)

    ## Create non-overlapping Crops for validation
    CreatCrops(args.base_dir,
                crop_type='test',
                size=args.crop_size,
                stride=args.crop_size,  ## Non-overlapping
                image_suffix=args.im_suffix,
                gt_suffix=args.gt_suffix)

    end = time.clock()
    print('Finished Creating crops, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()