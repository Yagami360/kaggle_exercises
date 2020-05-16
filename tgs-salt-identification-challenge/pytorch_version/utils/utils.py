# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import imageio
import random

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

#====================================================
# モデルの保存＆読み込み関連
#====================================================
def save_checkpoint(model, device, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)
    return

def save_checkpoint_w_step(model, device, save_path, step):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(
        {
            'step': step,
            'model_state_dict': model.cpu().state_dict(),
        }, save_path
    )
    model.to(device)
    return

def load_checkpoint(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    return

def load_checkpoint_w_step(model, device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint['step']
    model.to(device)
    return step



#====================================================
# 画像の保存関連
#====================================================
def save_image_w_norm( img_tsr, save_img_paths ):
    """
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5) 正規化した Tensor を画像として保存する。
    """
    img_tsr = (img_tsr.clone()+1)*0.5 * 255
    img_tsr = img_tsr.cpu().clamp(0,255)

    img_np = img_tsr.detach().numpy().astype('uint8')
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)
        img_np = img_np.swapaxes(0, 1).swapaxes(1, 2)
    elif img_np.shape[0] == 3:
        img_np = img_np.swapaxes(0, 1).swapaxes(1, 2)

    Image.fromarray(img_np).save(save_img_paths)
    return


def convert_rle(img, order='F', format=True):
    """
    画像を連長圧縮 [RLE : Run Length Encoding] にエンコードする。
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    returns run length as an array or string (if format is True)
    Source https://www.kaggle.com/bguberfain/unet-with-depth
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

#====================================================
# TensorBoard への出力関連
#====================================================
def tensor_for_board(img_tensor):
    # map into [0,1]
    tensor = (img_tensor.clone()+1) * 0.5
    tensor.cpu().clamp(0,1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1,3,1,1)

    return tensor

def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)
    
    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

    return canvas

def board_add_image(board, tag_name, img_tensor, step_count, n_max_images = 32):
    tensor = tensor_for_board(img_tensor)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)
    return

def board_add_images(board, tag_name, img_tensors_list, step_count, n_max_images = 32):
    tensor = tensor_list_for_board(img_tensors_list)
    tensor = tensor[0:min(tensor.shape[0],n_max_images)]
    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)
    return


#====================================================
# 評価指標関連
#====================================================
def iou_metric(y_true_in, y_pred_in, print_table=False):
    """
    IoU スコアを計算
    src: https://www.kaggle.com/aglotero/another-iou-metric

    [args]
        y_true_in : <ndarray> 正解画像
        y_pred_in : <ndarray> 予想画像
    """
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

#====================================================
# その他
#====================================================
def set_random_seed(seed=72):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return