import  lmdb
import pickle
import  numpy
#在进行OCR文本识别的过程中训练的数据量较大，所以采用将数据保存为LMDB数据，指定图片的主路径与标签文件
#图片的路径到标签 和特征的映射    就是一个字典

# env=lmdb.open("D:\\新建文件夹\\textvqa\\sam-textvqa\\data\\textVQA_datasets\\textvqa\\defaults\\features\\open_images\\detectron.lmdb",map_size=1099511627776)
# txn=env.begin(write=True)
# for key, value in txn.cursor():
#     print (type(key))

import lmdb

env_db = lmdb.Environment('../test_data/detectron.lmdb')
txn = env_db.begin(write=True)

#添加数据
# txn.put(key = '1'.encode(), value = 'aaa'.encode())
# txn.put(key = '2', value = 'bbb')
# txn.put(key = '3', value = 'ccc')
# txn.commit() #提交

# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None

# print
# txn.get(str(200))

# for key, value in txn.cursor():  # 遍历
#     print(key.decode(),value)

#key 查找
value=txn.get(str("test_task3/VisualGenome/1/712985").encode())   #vmb

print (pickle.loads(value))
env_db.close()


#vmb识别结果 npy
data=numpy.load("../test_data/000005.npy",allow_pickle=True)
data_info=numpy.load("../test_data/000005_info.npy",allow_pickle=True)
print(data.shape)   #(36,2048)的feature
print(data_info)    #其他的额外信息
