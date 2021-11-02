import  lmdb
import pickle
import array
import  numpy
#在进行OCR文本识别的过程中训练的数据量较大，所以采用将数据保存为LMDB数据，指定图片的主路径与标签文件
#图片的路径到标签 和特征的映射    就是一个字典


# key需要encode   读取frcnn  ocr
# env0=lmdb.open("../test_data/ocr_en_frcn_features.lmdb",map_size=10995116277)
# txn0=env0.begin(write=True)
# # for key,i in txn0.cursor():
# #     print(key)
# data=txn0.get("train/0013a4847289b1e5".encode())   #keys的value 也需要decode()
# da=pickle.loads(data)
# print(da)
# print(da["features"].shape)
# env0.close()



# # 读取 textvqa obj  lmdb
# env0=lmdb.open("../test_data/detectron.lmdb",map_size=8995116277)
# txn0=env0.begin(write=True)
# # for key,i in txn0.cursor():
# #     print(key)
# data=txn0.get("test_task3/VisualGenome/1/2349536".encode())   #keys的value 也需要decode()
# data=pickle.loads(data)
# print(data["features"][1].dtype)
# # print(data)
# env0.close()


# # 读取  est ocr
# env_db = lmdb.Environment('../test_data/MY_EST_CH_train_ocr_feature.lmdb')
# txn = env_db.begin(write=True)
# #添加数据
# # txn.put(key = '1'.encode(), value = 'aaa'.encode())
# # txn.put(key = '2', value = 'bbb')
# # txn.put(key = '3', value = 'ccc')
# # txn.commit() #提交
# # get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
# # txn.get(str(200))
# # i=0
# # for key, value in txn.cursor():  # 遍历
# #     print(key)
# #     i+=1
# # print(i)
# # key 查找
# # value=txn.get(str("008221").encode())
# #vmb   一共22025个
# value=txn.get(str("000000.npy").encode())
# print (pickle.loads(value))
# env_db.close()


# # 读取ocr
# env0=lmdb.open("../test_data/MY_EST_CH_train_ocr_feature.lmdb",map_size=1995116277)
# txn0=env0.begin(write=True)
# # for key,i in txn0.cursor():
# #     print(key,pickle.loads(i))
# #     if key != b"keys":
# #         fea=txn0.get(key)
# #         fea = pickle.loads(fea)
# #         if len(fea["features"])==1:
# #             print(fea["features"])
#         # print(fea["features"],key)
# data=txn0.get("000048.npy".encode())   #keys的value 也需要decode()   48 feature为空
# print(pickle.loads(data)["features"])
# print(pickle.loads(data))
# env0.close()




# 读取obj  EST
env0=lmdb.open("../test_data/tep/EST_train_obj_lmdb2.lmdb",map_size=3995116277)
txn0=env0.begin(write=True)
j=0
for i ,val in txn0.cursor():
    j+=1
print(j)
# txn0.delete("keys.npy".encode())
# txn0.commit()
data=txn0.get("000000.npy".encode())   #keys的value 也需要decode()
data=pickle.loads(data)
print(data)
env0.close()





# #修改obj的宽|高为float型
# env0=lmdb.open("../test_data/tep/EST_train_obj_lmdb2.lmdb",map_size=10995116277)
# txn0=env0.begin(write=True)
# j=0
# for i ,val in txn0.cursor():
#     # print(i)
#     j += 1
#     if j>=12000 and i!=b"keys":
#         print(i)
#         data=pickle.loads(val)
#         data["image_width"]=float(data["image_width"])
#         data["image_height"] = float(data["image_height"])
#         # print(data)
#
#         txn0.put(key=i,value=pickle.dumps(data))
#         if j==18000:
#             print(i)
#             break
#     if i==b"keys":
#         print(pickle.loads(val))
#         txn0.put(key=i,value=val)
# txn0.commit()
# #         # txn0.put(key=i,value=pickle.dumps(data))
# # # txn0.commit()
# # # data=txn0.get("000000.npy".encode())   #keys的value 也需要decode()
# # # print(pickle.loads(data)["features"].dtype)
# # # print(pickle.loads(data))
# env0.close()






# #用来查看
# #key需要encode
# env0=lmdb.open("../test_data/EST_train_obj_lmdb2.lmdb")
# txn0=env0.begin(write=True)
# # for key,i in txn0.cursor():
# #     print(key)
# data=txn0.get("000048.npy".encode())   #keys的value 也需要decode()
# data=pickle.loads(data)
# print(data["features"].tolist())
#
# print(data)
# env0.close()



# 用来修改  obj 键值为npy的形式
# env_db1= lmdb.Environment('../test_data/tep/EST_train_obj_lmdb2.lmdb',map_size=5995116277)
# env_db = lmdb.Environment('../test_data/tep/EST_train_obj_lmdb.lmdb',map_size=3089934592)
# txn = env_db.begin(write=True)
# txn1 = env_db1.begin(write=True)
# # 修改键值为.npy  且修改keys的值
# name_list=[]
# for key, value in txn.cursor():  # 遍历
#      # key 001457
#     # 将键值 修改为带npy的形式
#     if key!=b'keys':
#         new_key=""
#         new_key=key.decode()+".npy"
#         valu=pickle.loads(txn.get(key))
#         # print(key)
#         name=str(valu["feature_path"])
#         valu["feature_path"]=name+".npy"
#         name_list.append(new_key.encode())
#         txn1.put(key=new_key.encode(), value=value)
#     else:continue
# print(len(name_list))
# txn1.put(key=b"keys",value=pickle.dumps(name_list))
# txn1.commit()  #对修改将进行保存
# print("finish")
# env_db.close()
# env_db1.close()



#删除英文部分图片
# k_li=[]
# for key ,v in txn1.cursor():
#     k_li.append(key)
# print(len(k_li),k_li[:10])
# for key,value in txn1.cursor():
#     if key not in key_li_ch:
#         txn1.delete(key=key)
# txn1.commit()


#修改feature 的类型


# value=txn.get(str("008221").encode())   #vmb   一共22025个
# print (pickle.loads(value))
# value=txn1.get("002229.npy".encode())   #vmb   一共22025个
# print (pickle.loads(value))
# env_db.close()
# env_db1.close()







# data=numpy.load("../test_data/EST_npy/MY_EST_train.npy",allow_pickle=True)



#vmb识别结果 npy
# data=numpy.load("../test_data/000005.npy",allow_pickle=True)
# data_info=numpy.load("../test_data/000005_info.npy",allow_pickle=True)
# print(data.shape)   #(36,2048)的feature
# print(data_info)    #其他的额外信息



# 修改ocr feature 的类型    list  转 arr
# key_li_ch=[]
# env_ocr=lmdb.Environment("../test_data/MY_EST_CH_train_ocr_feature.lmdb",map_size=2995116277)
# txn_ocr = env_ocr.begin(write=True)
# for  key,value in txn_ocr.cursor():
#     key_li_ch.append(key)
# key_li_ch=key_li_ch[:-1]
# print(key_li_ch[:10],len(key_li_ch))
# for q in key_li_ch:
#     am=pickle.loads(txn_ocr.get(q))
#     # print(am["features"])
#     feature=[]
#     for i in am["features"]:
#         feature.append(i.tolist())
#     feature=numpy.array(feature,dtype=numpy.float32)
#     am["features"]=feature
#     txn_ocr.put(key=q,value=pickle.dumps(am))
# data=txn_ocr.get("000000.npy".encode())
# print(pickle.loads(data))
# print(pickle.loads(data)['features'])
# txn_ocr.commit()
# env_ocr.close()





# # 修改ocr feature 为空的部分
# 查看ocr    key需要encode
# env0=lmdb.open("../test_data/MY_EST_CH_train_ocr_feature.lmdb",map_size=1995116277)
# txn0=env0.begin(write=True)
# for key,i in txn0.cursor():
#     # print(key,pickle.loads(i)["features"])
#     if key !=b"keys":
#         fea=txn0.get(key)
#         fea = pickle.loads(fea)
#         if not fea["features"].tolist():
#             fea["features"]=numpy.array([[1.0]*2048],dtype=numpy.float32)
#             numpy.asarray(fea["features"],dtype=numpy.float32)
#             # fea["features"].astype="float32" #修改类型
#             txn0.put(key=key,value=pickle.dumps(fea))
# data=txn0.get("000048.npy".encode())
# data=pickle.loads(data)
# print(data)
# print(data["features"])#keys的value 也需要decode()
# # if  not data["features"].tolist():
# #     print("Kong!!!")
# # print(data)
# txn0.commit()
# env0.close()




# #ocr每个feature float转double
# env0=lmdb.open("../test_data/tep/MY_EST_CH_train_ocr_feature.lmdb",map_size=2995116277)
# txn0=env0.begin(write=True)
# for key,value in txn0.cursor():
#     if key==b"keys"  or key==b"keys.npy":
#         txn0.put(key=key,value=value)
#         continue
#     if len(key.decode())<20:
#         data=pickle.loads(value)
#         new_list=[]
#         print(key.decode())
#         for i in data["features"]:
#             #通过numpy将list转为double
#             numpy_feature=numpy.asarray(i,dtype=numpy.float64)
#             # print(numpy_feature.dtype)
#             new_list.append(numpy_feature.tolist())
#         data["features"]=numpy.array(new_list,dtype=numpy.float64)
#         txn0.put(key=key,value=pickle.dumps(data))
# da=txn0.get("000005.npy".encode())
# print(pickle.loads(da)["features"].dtype)
# print(pickle.loads(da)["features"][1].dtype)
# txn0.commit()
# env0.close()

