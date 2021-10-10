from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import os,glob


"""
利用pytorch.model.detect 里面封装的几个模型识别器进行识别
已调通，只能识别91种类别  效果一般般  原文地址
  https://blog.csdn.net/u013679159/article/details/104288338?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-5.no_search_link&spm=1001.2101.3001.4242
"""


# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

#训练的类别名称
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold=0.5):
    """

    :param img_path:   图片的路径
    :param threshold:  默认阈值为0.5
    :return: box和标签
    """

    img = Image.open(img_path) # Load the image
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img]) # Pass the image to the model
    print('pred')
    print(pred)
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    print("original pred_class")
    print(pred_class)
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    print("original pred_boxes")
    print(pred_boxes)
    pred_score = list(pred[0]['scores'].detach().numpy())
    print("orignal score")
    print(pred_score)
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    print(pred_t)
    print(pred_boxes)
    print(pred_class)
    return pred_boxes, pred_class
# boxes, pred_cls = get_prediction("D:\\新建文件夹\\daat_process\\pic_analsys\\OCR_BOX_Extract\\000015.jpg") # Get predictions


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    """
    调用检测函数  并使用opencv绘制包围框和label文本
    :param img_path:   本地路径不能包含中文
    :param threshold:
    :param rect_th:
    :param text_size:
    :param text_th:
    :return:
    """
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    index=0
    print("boxwei：",boxes)
    for box in boxes: #坐标不能有小数
        startX=int(box[0][0])
        startY=int(box[0][1])
        endX=int(box[1][0])
        endY=int(box[1][1])

        cv2.rectangle(img, (startX, startY), (endX, endY), color=(0, 255, 0),
                      thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[index], (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)  # Write the prediction class
        index+=1
    plt.figure(figsize=(5, 8))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def  get_filelist(dir):
    files_list = []
    x = os.walk(dir)
    for path, d, filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))

    print("文件夹包含文件为：",files_list)
    return files_list
dir="D:\\Download\\sample_picture\\pictures"
filelist=get_filelist(dir)
for i in filelist:
    object_detection_api(i)

