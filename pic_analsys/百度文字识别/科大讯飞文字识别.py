"""
  印刷文字识别WebAPI接口调用示例接口文档(必看)：https://doc.xfyun.cn/rest_api/%E5%8D%B0%E5%88%B7%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB.html
  上传图片base64编码后进行urlencode要求base64编码和urlencode后大小不超过4M最短边至少15px，最长边最大4096px支持jpg/png/bmp格式
  (Very Important)创建完webapi应用添加合成服务之后一定要设置ip白名单，找到控制台--我的应用--设置ip白名单，如何设置参考：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=41891
  错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
  @author iflytek
"""
# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import pickle
import json
import lmdb
import os, glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append("..")
from pic_feature_extract.vgg_extract_feature  import extractor_imgcode,Encoder

from PIL import Image

from tqdm import tqdm
"""效果最好的,对竖着的差"""

# from urllib import parse
# 印刷文字识别 webapi 接口地址
URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/general"
# 应用ID (必须为webapi类型应用，并印刷文字识别服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
APPID = "0c293935"
# 接口密钥(webapi类型应用开通印刷文字识别服务后，控制台--我的应用---印刷文字识别---服务的apikey)
API_KEY = "2994a6604d85311a34dcb8ce0d28ce41"

net=Encoder()
#调用请求头  修改调用参数
def getHeader(if_need_location):
    #  当前时间戳
    curTime = str(int(time.time()))
    #  支持语言类型和是否开启位置定位(默认否)
    param = {"language": "cn|en", "location": if_need_location}
    param = json.dumps(param)
    paramBase64 = base64.b64encode(param.encode('utf-8'))

    m2 = hashlib.md5()
    str1 = API_KEY + curTime + str(paramBase64, 'utf-8')
    m2.update(str1.encode('utf-8'))
    checkSum = m2.hexdigest()
    # 组装http请求头
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    return header

def  get_all_pic(data_dir):
    img_list = [os.path.join(data_dir, nm) for nm in os.listdir(data_dir) if nm[-3:] in ['jpg', 'png', 'gif']]
    img_list.sort()
    return  img_list



 #传入左上和右下坐标 进行描框
def  plot_box_in_picture(picpath,boxes):
    img = cv2.imread(picpath)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # print(img.shape)
    img1=Image.open(picpath)
    index = 0
    id=0
    featurelist=[]
    print("size为：", img1.size)
    # 坐标不能有小数
    for box in boxes:
        startX = int(box["location"]["top_left"]["x"])
        startY = int(box["location"]["top_left"]["y"])
        endX = int(box["location"]["right_bottom"]["x"])
        endY = int(box["location"]["right_bottom"]["y"])
        print("box为：",(startX, startY), (endX, endY))
        if startX>endX:startX,endX=endX,startX
        if startY > endY: startY, endY = endY, startY
        if startY ==endY: endY+=1
        if startX==endX:endX+=1

        cv2.rectangle(img, (startX, startY), (endX, endY), color=(0, 255, 0),
                      thickness=2)  # Draw Rectangle with the coordinates

        # 开始截取   用来对于ocr box  feature
        id+=1
        region = img1.crop((startX,startY,endX,endY))
        # 保存图片
        # region.save("./pic/test{}.jpg".format(id))

        # plt.imshow(region)
        # plt.show()
        #直接将region 送入到vgg  extractor 中
        #1channel  to 3 channel   pil
        im1=im2=im3=region.convert("L")
        region=Image.merge("RGB",(im1,im2,im3))
        feature=extractor_imgcode(region,net=net)
        # feature=[round(i, 8) for i in feature]
        featurelist.append(feature)


  #是否显示识别结构图片
    # plt.figure(figsize=(5, 8))  # display the LMDB_output image
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    return featurelist


def detet_a_pic_and_plot(pic_path):
    # 上传文件并进行base64位编码
    with open(pic_path, 'rb') as f:
        f1 = f.read()
    f1_base64 = str(base64.b64encode(f1), 'utf-8')
    f.close()

    data = {
        'image': f1_base64
    }
    r = requests.post(URL, data=data, headers=getHeader(if_need_location="true"))
    result = str(r.content, 'utf-8')
    # 错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)

    # 解析返回结果
    result_json = json.loads(result)


    #识别的文字结果和对应的精准位置
    # result_json['data']["block"][0]["line"][1]["word"][0]["location"] 是文字行的结果
    word_box_list = []
    text_detect=result_json['data']["block"][0]["line"]
    for i in text_detect:
        for j in i["word"]:
            word_box_list.append(j)
    # print("图片{a}的识别结果为:".format(a=pic_path),word_box_list)
    #描框
    feature_list=plot_box_in_picture(pic_path,word_box_list)   #  要描图此处取消注释

    return  word_box_list,feature_list



if __name__ == '__main__':


    # 识别a example
    pic_path="../test_data/020372.jpg"
    result,feature=detet_a_pic_and_plot(pic_path)
    print(result,feature)

    # # 识别本地样例图片并 进行描图
    def fun():
        pic_list=get_all_pic("D:\\Download\\sample_picture\\train")
        print("识别图片的数量为：",len(pic_list))
        res=[]
        feature_dictionary={}
        env_db = lmdb.Environment('../test_data/MY_EST_CH_train_ocr_feature.lmdb',map_size=2089934592)
        txn = env_db.begin(write=True)
        try:
            for i in pic_list:
                pic_name=os.path.basename(i)
                print(pic_name)
                result,feature_list=detet_a_pic_and_plot(i)
                res.append(result)
                feature_dictionary[pic_name]=feature_list
            # print(feature_dictionary)
            # print("该图片的特征列表为：",feature_list)
            # print(feature_list[0][:100])
            # print(result)
            # input("按任意键继续执行")
        except: #异常执行
            np.save("./MY_EST_CH_train_ocr_feature6.npy",feature_dictionary)
        else:
            np.save("./MY_EST_CH_train_ocr_feature6.npy",feature_dictionary)


        # except:
        #     env_db = lmdb.Environment('../test_data/MY_EST_CH_all_train_ocr_feature.lmdb')
        #     txn = env_db.begin(write=True)
        #     for key,value in feature_dictionary.items():
        #         value={"image_id":key,"features":value,"bbox":None,"num_boxes":None,
        #                "image_height":None,"image_width":None,
        #             "objects":None,"cls_prob":None}
        #         txn.put(key=key.encode(), value=pickle.dumps(value))
        #     txn.commit()  # 提交
        #     env_db.close()
        #
        # #保存并显示结果
        # else:
        #     env_db = lmdb.Environment('../test_data/MY_EST_CH_all_train_ocr_feature.lmdb')
        #     txn = env_db.begin(write=True)
        #     for key, value in feature_dictionary.items():
        #         value = {"image_id": key, "features": value, "bbox": None, "num_boxes": None,
        #                  "image_height": None, "image_width": None,
        #                  "objects": None, "cls_prob": None}
        #         txn.put(key=key.encode(), value=pickle.dumps(value))
        #     txn.commit()  # 提交
        #     env_db.close()
    fun()




    # 识别中文部分所有图片并保存特征 为json文件
    # pic_list = get_all_pic("D:\\Download\\sample_picture\\train2")
    #
    # print("识别图片的数量为：", len(pic_list))  #9515
    # res = []
    # print("*"*10,"detecting.....","*"*10)
    # for i in tqdm(pic_list):
    #     pic_name=os.path.basename(i)
    #     print(pic_name)
    #     result = detet_a_pic_and_plot(i)
    #     res.append({str(pic_name):result})
    #     # time.sleep(0.5)
    # with open('D:\\Download\\sample_picture\\中文训练集部分ocr结果2.json',"w",encoding="utf8") as f:
    #     json.dump(res,f,ensure_ascii=False)
    # print("识别结束")

