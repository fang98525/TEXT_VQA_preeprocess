"""
读取不同的实验需要的数据文件  从MMF源码里面copy的读取格式
"""
import numpy as np
from iopath.common.file_io import PathManager
import json
import datetime

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

data_textvqa=np.load("D:\\新建文件夹\\数据集处理\\textVQA数据集\\textvqa\\defaults\\annotations\\imdb_val_ocr_en.npy",allow_pickle=True)
print(data_textvqa[:2])

with open ("./test_data/ESTVQA_train.json","r") as f :
    data_EST=json.load(f)
    #在第一行插入创建信息
    curtime = datetime.datetime.now().strftime('%Y-%m-%d')
    data_EST.insert(0,{'creation_time':curtime, 'version': 0.5, 'dataset_type': 'ocr_lmdb_file', 'has_answer': True})
    # data_EST=data_EST[:2]
    print(data_EST[:2],len(data_EST))
with open("./test_data/EST_CH_OCR.json",'r',encoding="utf8") as f1 :
    ocr_list = []
    for line in f1.readlines():
        ocr_data=json.loads(line)#读取文件用load
        #按图片名字排序

        for item in ocr_data:
            ocr_list.append(item)
    print(ocr_list[:10],len(ocr_list))







