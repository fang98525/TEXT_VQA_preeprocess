from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import os
import cv2
"""
使用EAST进行scene text 检测      目录路径不能有中文

已调通   但是效果不太好   40张识别出来一半  识别出来中文的效果一般般   ,英文的效果还不错
"""
# construct the argument parser and parse the arguments

def   detect_one_picture(imagepath):
    width = 320
    height = 320  #  w和h必须为32 的倍数    resized image width (should be multiple of 32)
    min_confidence = 0.5   #minimum probability required to inspect a region     box的最小概率   提高精确度
    modelpath = "D:\\Download\\EAST_model\\frozen_east_text_detection.pb"   #east预训练模型
    save_dir="D:\\Download\\sample_picture\\output"

    # load the input image and grab the image dimensions
    image = cv2.imread(imagepath)
    # cv2.imshow("iamage",image)
    # cv2.waitKey(0)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = W / float(newW)    #scale=原始size ：resize的 320
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two LMDB_output layer names for the EAST detector model that
    # we are interested -- the first is the LMDB_output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(modelpath)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two LMDB_output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)    #得到置信度  和box
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # 从scores volume中获取行数和列数，然后初始化我们的边界框矩形集和相应的置信度分数
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image     根据box描点
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the LMDB_output image
    cv2.imshow("Text Detection", orig)
    # cv2.imwrite("D:\\Download\\sample_picture\\LMDB_output", orig)  #保存识别的图片   暂时不保存
    cv2.waitKey(0)

def get_pict_pathlists(dir_path):
    filelist = os.listdir(dir_path)
    pathlist=[]
    for path in filelist:
        imagepath=dir_path+"\\"+path
        pathlist.append(imagepath)
    return  pathlist

if __name__ == '__main__':
    dirpath = "D:\\Download\\sample_picture\\pictures"
    pathlist=get_pict_pathlists(dir_path=dirpath)
    print(pathlist)
    for path in pathlist:
        detect_one_picture(path)
    # detect_one_picture("./013388.jpg")#   英文的检测效果比较好

