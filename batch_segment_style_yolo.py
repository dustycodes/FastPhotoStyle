"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import argparse
import glob
import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.misc import imread, imresize
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import process_stylization_ade20k_ssn
from photo_wct import PhotoWCT
from segmentation.mit_semseg.dataset import round2nearest_multiple
from segmentation.mit_semseg.lib.nn import (async_copy_to,
                                            user_scattered_collate)
from segmentation.mit_semseg.lib.utils import as_numpy, mark_volatile
from segmentation.mit_semseg.models import ModelBuilder, SegmentationModule

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model_path', help='folder to model path', default='ade20k-resnet50dilated-ppm_deepsup')
parser.add_argument('--suffix', default='_epoch_20.pth', help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet50_dilated8', help="architecture of net_encoder")
parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup', help="architecture of net_decoder")
parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')
parser.add_argument('--num_val', default=-1, type=int, help='number of images to evalutate')
parser.add_argument('--num_class', default=150, type=int, help='number of classes')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize. current only supports 1')
parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')
parser.add_argument('--imgMaxSize', default=1000, type=int, help='maximum input image size of long edge')
parser.add_argument('--padding_constant', default=8, type=int, help='maxmimum downsampling rate of the network')
parser.add_argument('--segm_downsampling_rate', default=8, type=int, help='downsampling rate of the segmentation label')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')

parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth', help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--content_image_path', default="./images/content3.png")
parser.add_argument('--content_image_dir', default="./content")
parser.add_argument('--content_seg_path', default='./results/content3_seg.pgm')
parser.add_argument('--style_image_path', default='./images/style3.png')
parser.add_argument('--style_image_dir', default='./styles')
parser.add_argument('--style_seg_path', default='./results/style3_seg.pgm')
parser.add_argument('--output_image_path', default='./results/example3.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--output_visualization', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument('--label_mapping', type=str, default='ade20k_semantic_rel.npy')
parser.add_argument('--results_dir', type=str, default='/media/dusty/Storage1/yolo-style')
parser.add_argument('--percent_resized_style', type=str, default='50')
parser.add_argument('--percent_resized_content', type=str, default='50')
parser.add_argument('--percent_upscaled_result', type=str, default='200')
args = parser.parse_args()

segReMapping = process_stylization_ade20k_ssn.SegReMapping(args.label_mapping)

# Absolute paths of segmentation model weights
SEG_NET_PATH = 'segmentation'
args.weights_encoder = os.path.join(SEG_NET_PATH, "ckpt", args.model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(SEG_NET_PATH, "ckpt", args.model_path, 'decoder' + args.suffix)
args.arch_encoder = 'resnet50dilated'
args.arch_decoder = 'ppm_deepsup'
# args.arch_decoder = 'ppm_bilinear_deepsup'
args.fc_dim = 2048

# Load semantic segmentation network module
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])

# Load FastPhotoStyle model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))
if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(0)


def segment_this_img(f):
    img = imread(f, mode='RGB')
    img = img[:, :, ::-1]  # BGR to RGB!!!
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in args.imgSize:
        scale = this_short_size / float(min(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, args.padding_constant)
        target_width = round2nearest_multiple(target_width, args.padding_constant)
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input = dict()
    input['img_ori'] = img.copy()
    input['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (img.shape[0],img.shape[1])
    with torch.no_grad():
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            feed_dict = dict()
            feed_dict['img_data'] = timg.cuda()
            feed_dict = async_copy_to(feed_dict, args.gpu_id)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp.cpu() / len(args.imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))
    return preds


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


style_files = [f for f in glob.glob(f"{args.style_image_dir}/*.jpg")]
content_files = [f for f in glob.glob(f"{args.content_image_dir}/**/*.jpg", recursive=True)]
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

style_num = 0
content_num = 0
for style_image in tqdm(style_files, desc="Going through content style images"):
    for content_image in tqdm(content_files, desc="Going through content images"):
        try:
            results_dir = f"{args.results_dir}/style-{style_num}-content-{content_num}"
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)

            content_segment_image = f"{results_dir}/content_seg.pgm"
            style_segment_image = f"{results_dir}/style_seg.pgm"
            output_image_path = f"{results_dir}/result-{style_num}-{content_num}.png"
            output_image_path_upscaled = f"{results_dir}/result-{style_num}-{content_num}-upscaled.png"
            output_bbox_image_path_upscaled = f"{results_dir}/result-bbox-{style_num}-{content_num}-upscaled.png"
            content_resized_image_path = f"{results_dir}/content_resized.png"
            style_resized_image_path = f"{results_dir}/style_resized.png"
            yolo_file = content_image.replace(content_image.split(".")[-1], "txt")
            yolo_file_new = f"{output_image_path_upscaled.split('/')[-1].split('.')[:-1]}.txt".replace("[", "").replace("]", "")

            if os.path.exists(output_bbox_image_path_upscaled):
                continue

            os.system(f"convert -resize {args.percent_resized_style}% {style_image} {style_resized_image_path}")
            os.system(f"convert -resize {args.percent_resized_style}% {content_image} {content_resized_image_path}")
            os.system(f"cp {style_image} {results_dir}/{style_image.split('/')[-1]}")
            os.system(f"cp {content_image} {results_dir}/{content_image.split('/')[-1]}")

            cont_seg = segment_this_img(content_resized_image_path)
            cv2.imwrite(content_segment_image, cont_seg)
            style_seg = segment_this_img(style_resized_image_path)
            cv2.imwrite(style_segment_image, style_seg)
            process_stylization_ade20k_ssn.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_resized_image_path,
                style_image_path=style_resized_image_path,
                content_seg_path=content_segment_image,
                style_seg_path=style_segment_image,
                output_image_path=output_image_path,
                cuda=True,
                save_intermediate=args.save_intermediate,
                no_post=args.no_post,
                label_remapping=segReMapping,
                output_visualization=args.output_visualization
            )
            with Image.open(content_image) as og_im:
                og_width, og_height = og_im.size

            with Image.open(output_image_path) as res_im:
                d = res_im.resize((og_width, og_height), resample=Image.BILINEAR)
                d.save(output_image_path_upscaled)
            
            os.system(f"cp {yolo_file} {results_dir}/{yolo_file_new}")
            with open(yolo_file) as file:
                yolo_data = file.read()

            yolo_data = yolo_data.split()
            x_center, y_center, w, h = float(yolo_data[1])*og_width, float(yolo_data[2])*og_height, float(yolo_data[3])*og_width, float(yolo_data[4])*og_height
            x1 = round(x_center-w/2)
            y1 = round(y_center-h/2)
            x2 = round(x_center+w/2)
            y2 = round(y_center+h/2)
            bbox_image = cv2.imread(output_image_path_upscaled)
            plot_one_box([x1,y1,x2,y2], bbox_image, color=[255, 0, 0], label="drone", line_thickness=None)

            cv2.imwrite(output_bbox_image_path_upscaled, bbox_image) 

            content_num += 1
        except Exception as ex:
            print(ex)
            content_num += 1
    
    style_num += 1
