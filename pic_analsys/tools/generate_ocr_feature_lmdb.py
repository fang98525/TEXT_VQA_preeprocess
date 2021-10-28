import numpy
import lmdb
import pickle
#训练集的
data={}
feature_dict0=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature.npy",allow_pickle=True)
feature_dict1=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature1.npy",allow_pickle=True)
feature_dict2=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature2.npy",allow_pickle=True)
feature_dict3=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature3.npy",allow_pickle=True)
feature_dict4=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature4.npy",allow_pickle=True)
feature_dict5=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature5.npy",allow_pickle=True)
feature_dict6=numpy.load("../百度文字识别/MY_EST_CH_train_ocr_feature6.npy",allow_pickle=True)
feature_dict0.item().update(feature_dict1.item())
feature_dict0.item().update(feature_dict2.item())
feature_dict0.item().update(feature_dict3.item())
feature_dict0.item().update(feature_dict4.item())
feature_dict0.item().update(feature_dict5.item())
feature_dict0.item().update(feature_dict6.item())
# print(feature_dict0)
feature_dict0.item()
print(feature_dict0.item()["000000.jpg"])
print(len(feature_dict0.item().keys()))
env_db = lmdb.Environment('../test_data/MY_EST_CH_train_ocr_feature.lmdb',map_size=3089934592)
txn = env_db.begin(write=True)
key_list=[]
for i in feature_dict0.item().keys():
    value=feature_dict0.item()[i]
    name=i.replace(".jpg",".npy")
    value = {"image_id": name, "features": value}
    key_list.append(name.encode())
    txn.put(key = name.encode(), value = pickle.dumps(value))
print(key_list[:10])
txn.put(key = b"keys", value = pickle.dumps(key_list))
txn.commit()
env_db.close()