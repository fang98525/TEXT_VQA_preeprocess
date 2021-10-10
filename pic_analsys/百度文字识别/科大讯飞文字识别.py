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
import json
import os, glob
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
"""效果最好的,对竖着的差"""

# from urllib import parse
# 印刷文字识别 webapi 接口地址
URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/general"
# 应用ID (必须为webapi类型应用，并印刷文字识别服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
APPID = "2b923e8a"
# 接口密钥(webapi类型应用开通印刷文字识别服务后，控制台--我的应用---印刷文字识别---服务的apikey)
API_KEY = "972af91063fdd506f39a94fd0f13fa43"


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
    files =os.listdir(data_dir)
    files_list=[]
    for filename in files:
        file_glob = os.path.join(data_dir, filename)
        files_list.extend(glob.glob(file_glob))
    return  files_list


#传入左上和右下坐标 进行描框
def  plot_box_in_picture(picpath,boxes):
    img = cv2.imread(picpath)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    index = 0
    # 坐标不能有小数
    for box in boxes:
        startX = int(box["location"]["top_left"]["x"])
        startY = int(box["location"]["top_left"]["y"])
        endX = int(box["location"]["right_bottom"]["x"])
        endY = int(box["location"]["right_bottom"]["y"])

        cv2.rectangle(img, (startX, startY), (endX, endY), color=(0, 255, 0),
                      thickness=2)  # Draw Rectangle with the coordinates


    plt.figure(figsize=(5, 8))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


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
    print(result_json)

    #识别的文字结果和对应的精准位置
    # result_json['data']["block"][0]["line"][1]["word"][0]["location"] 是文字行的结果
    word_box_list = []
    text_detect=result_json['data']["block"][0]["line"]
    for i in text_detect:
        for j in i["word"]:
            word_box_list.append(j)
    print("图片{a}的识别结果为:".format(a=pic_path),word_box_list)
    #描框
    # plot_box_in_picture(pic_path,word_box_list)   #  要描图此处取消注释

    return  word_box_list



if __name__ == '__main__':

    # 识别a example
    # pic_path="../test_data/000018.jpg"
    # result=detet_a_pic_and_plot(pic_path)

    #识别本地样例图片并 进行描图
    # pic_list=get_all_pic("D:\\Download\\sample_picture\\pictures")
    # print("识别图片的数量为：",len(pic_list))
    # res=[]
    # for i in pic_list:
    #     result=detet_a_pic_and_plot(i)
    #     res.append(result)

    # 识别中文部分所有图片并保存特征
    pic_list = get_all_pic("D:\\Download\\sample_picture\\train")
    print("识别图片的数量为：", len(pic_list))  #9515
    res = []
    print("*"*10,"detecting.....","*"*10)
    for i in tqdm(pic_list):
        result = detet_a_pic_and_plot(i)
        res.append(result)
    f = open('D:\\Download\\sample_picture\\train\\中文训练集部分ocr结果.txt', mode='w+')
    f.write(res)
    f.close()
    print("识别结束")

