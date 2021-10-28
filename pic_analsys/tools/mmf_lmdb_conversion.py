# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import os
import pickle

import lmdb
import numpy as np
import tqdm
from iopath.common.file_io import PathManager as pm


PathManager = pm()

"""将lmdb里面存储的npy  以及npy  info  还原出来
以及将提取npy————》lmdb
  """
class LMDBConversion:
    def __init__(self):
        self.args = self.get_parser().parse_args()

    def get_parser(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "--mode",

            type=str,
            default="extract",
            help="Mode can either be `convert` (for conversion of \n"
            + "features to an LMDB file) or `extract` (extract \n"
            + "raw features from a LMDB file)",
        )
        parser.add_argument(
            "--lmdb_path",  type=str, help="LMDB file path"
            ,default="../test_data/ocr_en_frcn_features.lmdb"
        )
        parser.add_argument(
            "--features_folder", type=str, help="Features folder",
            default="../test_data/detectron.lmdb/LMDB_output"
        )
        return parser

#将npy 文件存到 lmdb字典
    def convert(self):
        env = lmdb.open(self.args.lmdb_path, map_size=1099511627776)
        id_list = []
        #所有的npy文件 的绝对路径
        all_features = glob.glob(
            os.path.join(self.args.features_folder, "**", "*.npy"), recursive=True
        )
      #所有 feature npy文件的绝对路径
        features = []
        #读取npy feature list     key就是path
        for feature in all_features:
            if not feature.endswith("_info.npy"):
                features.append(feature)

        with env.begin(write=True) as txn:
            for infile in tqdm.tqdm(features):
                reader = np.load(infile, allow_pickle=True)
                item = {}
                #或取文件的纯名称  0000005  也是lmdb的key
                split = os.path.relpath(infile, self.args.features_folder).split(
                    ".npy"
                )[0]
                item["feature_path"] = split
                key = split.encode()
                id_list.append(key)
                item["features"] = reader #1
                info_file = infile.split(".npy")[0] + "_info.npy"
                if not os.path.isfile(info_file):
                    txn.put(key, pickle.dumps(item))
                    continue

                reader = np.load(info_file, allow_pickle=True)
                item["image_height"] = reader.item().get("image_height")
                item["image_width"] = reader.item().get("image_width")
                item["num_boxes"] = reader.item().get("num_boxes")
                item["objects"] = reader.item().get("objects")
                item["cls_prob"] = reader.item().get("cls_prob", None)
                item["bbox"] = reader.item().get("bbox")#7

                txn.put(key, pickle.dumps(item))

            txn.put(b"keys", pickle.dumps(id_list))

    def extract(self):
        os.makedirs(self.args.features_folder, exist_ok=True)
        env = lmdb.Environment(
            self.args.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            i=0
            _image_ids = pickle.loads(txn.get(b"keys"))
            for img_id in tqdm.tqdm(_image_ids):
                item = pickle.loads(txn.get(img_id))
                img_id = img_id.decode("utf-8")
                tmp_dict = {}
                tmp_dict["image_id"] = img_id
                tmp_dict["bbox"] = item["bbox"]
                tmp_dict["num_boxes"] = item["num_boxes"]
                tmp_dict["image_height"] = item["image_height"]
                tmp_dict["image_width"] = item["image_width"]
                tmp_dict["objects"] = item["objects"]
                tmp_dict["cls_prob"] = item["cls_prob"]

                info_file_base_name = str(img_id) + "_info.npy"
                file_base_name = str(img_id) + ".npy"

                path = os.path.join(self.args.features_folder, file_base_name)
                if PathManager.exists(path):
                    continue
                info_path = os.path.join(self.args.features_folder, info_file_base_name)
                base_path = "/".join(path.split("/")[:-1])
                PathManager.mkdirs(base_path)
                np.save(PathManager.open(path, "wb"), item["features"])
                np.save(PathManager.open(info_path, "wb"), tmp_dict)
                # i+=1
                # if i==10:
                #     break

    def execute(self):
        if self.args.mode == "convert":
            self.convert()
        elif self.args.mode == "extract":
            self.extract()
        else:
            raise ValueError("mode must be either `convert` or `extract` ")


if __name__ == "__main__":
    lmdb_converter = LMDBConversion()
    lmdb_converter.execute()
