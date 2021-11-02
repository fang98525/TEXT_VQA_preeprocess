"""
读取不同的实验需要的数据文件  从MMF源码里面copy的读取格式
"""
import numpy as np
from iopath.common.file_io import PathManager
import json
import datetime
from PIL import Image
import lmdb
import pickle



#mmf 读取 annoation   可以自己修改
def _load_jsonl( path):
    with PathManager.open(path, "r") as f:
        db = f.readlines()
        for idx, line in enumerate(db):
            db[idx] = json.loads(line.strip("\n"))
        data = db
        start_idx = 0


def _load_npy( path):
    with open(path, "rb") as f:
        db = np.load(f, allow_pickle=True)

    start_idx = 0

    if type(db) == dict:
        metadata = db.get("metadata", {})
        data = db.get("data", [])
    else:
        # TODO: Deprecate support for this
        metadata = {"version": 1}
        data = db
        # Handle old imdb support
        if "image_id" not in data[0]:
            start_idx = 1

    if len(data) == 0:
        data = db
    return db


def _load_json( path):
    with PathManager.open(path, "r") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    data = data.get("data", [])

    if len(data) == 0:
        raise RuntimeError("Dataset is empty")


def get_pic_size(pic_dir):
    img = Image.open(pic_dir)
    w,h=img.size
    # print(w,h)
    # print(img.format)
    return w,h
# get_pic_size("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\images\\train\\000000.jpg")

def load_annotation_db(path):

    if path.endswith(".npy"):
         return _load_npy(path)
    elif path.endswith(".jsonl"):
        _load_jsonl(path)
    elif path.endswith(".json"):
        _load_json(path)
    else:
        raise ValueError("Unknown file format for annotation db")
# data_textvqa=load_annotation_db("D:\\新建文件夹\\数据集处理\\textVQA数据集\\textvqa\\defaults\\annotations\\imdb_val_ocr_en.npy")
# print(data_textvqa[:2])

data_textvqa=np.load("D:\\新建文件夹\\数据集处理\\textVQA_datasets\\textvqa\\defaults\\annotations\\imdb_val_ocr_en.npy",allow_pickle=True)
print("textVQA npy:",data_textvqa[:2])  #需要靠近的形式


#科大讯飞提取的ocr信息
with open("test_data/EST_CH_OCR_train.json", 'r', encoding="utf8") as f1 :
    ocr_list = []
    for line in f1.readlines():
        ocr_data=json.loads(line)#读取文件用load
        #按图片名字排序

        for item in ocr_data:
            ocr_list.append(item)
    print("EST_CH_OCR_train.json的信息：",ocr_list[:1],"\n",len(ocr_list))




#读取EST　的ｏｂｊ　ｂｏｘ　和ｆｅａｔｕｒｅ
def get_LMDB_data(name):
    env_db = lmdb.Environment('./test_data/EST_train_obj_lmdb')
    txn = env_db.begin(write=True)
    value=txn.get(str(name).encode())   #vmb   一共22025个
    # print (pickle.loads(value))
    LMDB_data=pickle.loads(value)
    env_db.close()
    return  LMDB_data

# env_db = lmdb.Environment('./test_data/EST_train_obj_lmdb.lmdb')
# txn = env_db.begin(write=True)
# value=txn.get(str("000000").encode())   #vmb   一共22025个
# print (pickle.loads(value))
# env_db.close()




#得到数据集的npy文件
with open ("./test_data/EST_CH_train.json","r",encoding="utf8") as f :
    data_EST=json.load(f)
    #在第一行插入创建信息
    curtime = datetime.datetime.now().strftime('%Y-%m-%d')
    data_EST.insert(0,{'creation_time':curtime, 'version': 0.5, 'dataset_type': 'val', 'has_answer': True})

    #获取所有的名称lsit    image path 就是image name
    name_list=[]
    for  sample in data_EST[1:]:
        name_list.append(sample["image"])
    new_sample_list=[]
    index = 0
    for i in data_EST[1:]:
        # 只截取第一个问题   a pic a question
        for annoation in i["annotation"]:
            i["annotation"]=annoation
            break
        i["question"]= i["annotation"]['question']
        i['image_id']=i["image"].replace(".jpg","")  #为图片名
        pic_path="D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\images\\train\\"+i["image"]
        width,height=get_pic_size(pic_path)
        i['image_width'],i["image_height"]=width,height
        i["answers"]=[i["annotation"]['answer']  for  _ in range(10)]   # 暂时没有对答案添加干扰
        i['question_tokens']=[token  for token in   i["annotation"]['question'].replace("?","")   ]             #去掉？进行中文分字
        i["question_id"]=i["annotation"]['question_id']
        i["set_name"]="val"  #没有划分数据集  暂时都设为train
        i['image_name']= i['image_id']
        i['image_path']=i["image"]   #暂时取图片全名
        i['feature_path']=i['image_id']  #不需要npy
        i["valid_answers"]=[i["annotation"]['answer']  for  _ in range(10)]
        #加载ocr信息  一一对应的
        word_list ,ocr_box_list= [],[]
        for k in  ocr_list[index].values():
            for  j in k:
                # print(j["location"])
                word_list.append(j["content"])
                #根据ocr 信息计算 ocr  normalized  box
                startX=j["location"]['top_left']["x"]
                starty = j["location"]['top_left']["y"]
                endX=j["location"]['right_bottom']["x"]
                endy= j["location"]['right_bottom']["y"]
                box=[round(startX/width,16),round(starty/height,16),round(endX/width,16),round(endy/height,16)]
                ocr_box_list.append(box)
        # print(ocr_box_list)
        index+=1
        if  not word_list:   #对ocr结果为空的进行处理
            word_list.append("无")
            # print(i["image"])
            ocr_box_list.append([0.5]*4)
        i['ocr_tokens'] =word_list
        #加载ocrbox 信息
        i["ocr_normalized_boxes"]=np.array(ocr_box_list,dtype=np.float32)
        lmdb_data=get_LMDB_data(i["image_name"])
        L_w,L_h=lmdb_data["image_width"],lmdb_data["image_height"]
        L_box=[]
        for item in lmdb_data["bbox"]:
                box_L = [round(item[0] / L_w, 8), round(item[1] / L_h, 8), round(item[2]/ L_w, 8),round(item[3]/ L_h, 8)]
                L_box.append(box_L)
        i['obj_normalized_boxes']=np.array(L_box,dtype=np.float32)
        del i["annotation"]

    #前8000个为训练集  保存为json文件
    # with open('./test_data/EST_npy/MY_EST_train.json', "w", encoding="utf8") as f:
    #     json.dump(data_EST[:8001],f,ensure_ascii=False)
    # with open('./test_data/EST_npy/MY_EST_val.json', "w", encoding="utf8") as f:
    #     json.dump(data_EST[8001:],f,ensure_ascii=False)
    #保存为npy文件
    # np.save('./test_data/EST_npy/MY_EST_train.npy',data_EST[:8001])
    np.save('./test_data/EST_npy/MY_EST_val.npy', data_EST[8001:])   #要修改setname  为val
    print(data_EST[39],len(data_EST),len(name_list))

















