#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d
import itertools
import json

from vision_utils.logger import get_logger
logger=get_logger()

image_ext = ['jpg', 'jpeg', 'png']
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

def is_image(file_name):
  ext = file_name[file_name.rfind('.') + 1:].lower()
  return ext in image_ext

times = []
def demo_image(image, img_name, model, opt):
    with CodeTimer() as timer:
      s = max(image.shape[0], image.shape[1]) * 1.0
      c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
      trans_input = get_affine_transform(
          c, s, 0, [opt.input_w, opt.input_h])
      inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                             flags=cv2.INTER_LINEAR)
      inp = (inp / 255. - mean) / std
      inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
      inp = torch.from_numpy(inp).to(opt.device)
      out = model(inp)[-1]
      pred = get_preds(out['hm'].detach().cpu().numpy())[0]
      pred = transform_preds(pred, c, s, (opt.output_w, opt.output_h))

      pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
                             out['depth'].detach().cpu().numpy())[0]

    time = timer.took

    num_joints = pred.shape[0]
    points = ((pred.reshape(num_joints, -1))).astype(np.int32)
    keypoints = []
    for j in range(num_joints):
        keypoints.extend([int(points[j, 0]), int(points[j, 1]), 1])
    logger.debug(keypoints)
    json_out = [{'keypoints':keypoints}]
    json_out_name = '../../eval/pytorch-pose-hg-3d/' + img_name + '.predictions.json'
    with open(json_out_name, 'w') as f:
        json.dump(json_out, f)
    logger.info(json_out_name)

    debugger = Debugger(ipynb=True)
    debugger.add_img(image)
    result = debugger.add_point_2d(pred, (255, 0, 0))
    cv2.imwrite(img_name,result)
      #
      # debugger.add_img(image)
      # debugger.add_point_2d(pred, (255, 0, 0))
      # debugger.add_point_3d(pred_3d, 'b')
      # debugger.show_all_imgs(pause=False)
      # debugger.show_3d()
    return time

scale=1

import glob
import numpy
from vision_utils.timing import CodeTimer


def main(opt):
    opt.heads['depth'] = opt.num_output
    if opt.load_model == '':
        opt.load_model = '../models/fusion_3d_var.pth'
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
        else:
            opt.device = torch.device('cpu')

    model, _, _ = create_model(opt)
    model = model.to(opt.device)
    model.eval()

    for test_image in glob.glob(f"/home/robotlab/pose test input/*.png"):
        img_name = f'{test_image.split("/")[-1].split(".")[-2]}-{scale}.{test_image.split(".")[-1]}' if scale<1 else test_image.split("/")[-1]
        image = cv2.imread(test_image)
        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, dim)
        times.append(demo_image(image, img_name, model, opt))
        logger.info(test_image)
    print(np.mean(times))


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
