import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import torchvision
from tqdm import tqdm
from mask_rcnn.utils import display_instances


if torch.cuda.is_available():
    torch.cuda.set_device(0)

data = glob.glob('/mnt/data/Public_datasets/COCO/train2017/000000000*')
data = data[10:20]

x = [torch.tensor(plt.imread(file)).permute(2,0,1).to(torch.float).cuda() / 255. for file in data]

weights = torch.load('mask_weights.pth')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.eval().cuda()


out = model(x)



# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

lines = open('mask_rcnn/coco_classes.txt', 'r').readlines()
class_names = [line.split('\n')[0] for line in lines]


for batch_idx in range(0,10):
    r = out[batch_idx]
    image = x[batch_idx].permute(1,2,0).detach().cpu().numpy()
    boxes = r['boxes'].detach().cpu().numpy()
    masks = r['masks'].permute(0,2,3,1)[:,:,:,0].detach().cpu().numpy()
    labels = r['labels'].detach().cpu().numpy()
    scores = r['scores'].detach().cpu().numpy()

    score_threshold = 0.2

    valid = scores > score_threshold

    boxes = boxes[valid]
    masks = masks[valid]
    labels = labels[valid]
    scores = scores[valid]

    list_mask = [i * 255 for i in masks]

    from vis import blend_img, COLORS_CLAZZ
    label_mask = np.zeros(masks[0].shape, dtype=int) # reserve -1

    for idx in range(len(masks)):
        tmp_mask = masks[idx] > score_threshold
        label_mask[tmp_mask] = idx + 1

    import cv2
    box_img = image.copy()
    thickness = 1
    font = cv2.QT_FONT_NORMAL
    fontScale = 0.4

    for idx in range(len(boxes)):
        color = tuple(COLORS_CLAZZ[idx, :3].astype(int))
        color = (int(color[2]), int(color[1]), int(color[0])) # BGR and int just because cv2

        start_point = (int(boxes[idx, 0]), int(boxes[idx, 1]))
        end_point = (int(boxes[idx, 2]), int(boxes[idx, 3]))

        box_img = cv2.rectangle(box_img, start_point, end_point, color, thickness)
        box_img = cv2.putText(box_img, class_names[labels[idx]] + '-' + f"{scores[idx]:.3f}", start_point, font,
                            fontScale, color, thickness, cv2.LINE_AA)

    out_img = blend_img(image, label_mask)
    plt.imshow(out_img)
    plt.savefig(f"tmp/{batch_idx}over.png")
    plt.imshow(box_img)
    plt.savefig(f"tmp/{batch_idx}box.png")

