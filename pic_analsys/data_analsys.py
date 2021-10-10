import numpy
import pandas
import lmdb
import pickle
import  json
import cv2
from PIL import Image
import os
import shutil #复制文件用


def  annoation_analys():
    """对训练集进行分析"""
    with open("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\annotations\\train.json")  as f:
        train=json.load(f,encoding="utf8")
        print(train[:100])

        train=pandas.DataFrame(data=train)
        pandas.set_option('display.max_columns', None)
        print(train.head())
        print(train.info())
        print(train["annotation"])
        index,total_question=0 ,0
        for i in train["annotation"] :
            print("图片：",train["image"][index],"索引是：",index)
            for annoation in i:
               print("问题是：" ,annoation["question"],"答案是：",annoation["answer"],"第{}个问题".format(total_question))
               total_question+=1
            index+=1



    data=numpy.load("D:\\新建文件夹\\数据集处理\\000005.npy",allow_pickle=True)
    print(data,data.shape)

    data=numpy.load("D:\\新建文件夹\\数据集处理\\000005_info.npy",allow_pickle=True)
    print("obj检测的信息以及size",data,data.shape)
    data=numpy.load("D:\\新建文件夹\\数据集处理\\m4c_data_sample\\annotations\\imdb_subval.npy",allow_pickle=True)
    i=23
    print("imdb_subval的信息以及size",data[:2],data[i]["obj_normalized_boxes"].shape,data[i]["ocr_normalized_boxes"].shape)
# annoation_analys()




"""获取dir下所有图片的size信息"""
def get_Allpics_size(dir_pah):
    filelist=os.listdir(dir_pah)
    for item in filelist:
        if item.endswith(".jpg"):
            path=dir_pah+item
            imag = Image.open(path)
            print("图片{}的尺寸为".format(item), imag.size)
# get_Allpics_size("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\images\\train\\")




def get_answer_list(train_json_dir):
    """

    :param train_json_dir:  训练集的地址
    :return:  答案的text list  以及question list
    """
    answer_list=[]
    question_list=[]
    with open(train_json_dir)  as f:
        train=json.load(f,encoding="utf8")
        train=pandas.DataFrame(data=train)
        pandas.set_option('display.max_columns', None)
        index,total_question=0 ,0
        for i in train["annotation"] :
            for annoation in i:
               question_list.append(annoation["question"])
               answer_list.append(annoation["answer"])
               # print("问题是：" ,annoation["question"],"答案是：",annoation["answer"],"第{}个问题".format(total_question))
               total_question+=1
            index+=1
    return  question_list ,answer_list
# question_list ,answer_list=get_answer_list("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\annotations\\train.json")
# print(question_list,answer_list)



def count_words(s, n):
    """返回字符串s中出现频率最高的n个词."""

    s_list = s.lower().split(' ')  # 单词统一转换为小写形式，并以空格进行切分

    # 统计字符串s中每个单词出现的次数
    top_n_dict = {}
    for word in s_list:
        if word in top_n_dict:
            top_n_dict[word] += 1
        else:
            top_n_dict[word] = 1

    # 按照出现频次对单词进行排序，如果出现频次相同，则按字母顺序排序
    word_frequency = []
    values = sorted(list(set(top_n_dict.values())), reverse=True)  # 统计所有单词出现的频次情况，将频次降序放入列表
    for w in values:
        # 将出现频次相同的单词放在一个列表里
        word_list = []
        for k, v in top_n_dict.items():
            if v == w:
                word_list.append((k, v))
        # 将出现频次相同的单词排序后添加到词频列表
        word_frequency.extend(sorted(word_list))

    # 返回出现频次排前n的单词
    return word_frequency[:n]
# print(count_words("Cat bat mat cat bat Cat", 3))
# print(count_words("betty bought a bit of Butter but the Butter was bitter", 3))


#检验字符串是否全为中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


#检验是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

#获取中文问题图片的名称
def get_chinese_pic_list_of_ques():
    # with open("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\annotations\\train.json")  as f:
    with open("/data1/home/fzq/projects/mmf/data/datsets/EST-VQA-v1.0/annotations/test.json")  as f:
        train=json.load(f,encoding="utf8")

        train = pandas.DataFrame(data=train)
        pandas.set_option('display.max_columns', None)
        index, total_question = 0, 0
        pic_list=[]
        for i in train["annotation"] :
            print("图片：", train["image"][index], "索引是：", index)
            for annoation in i:
               print("问题是：", annoation["question"], "答案是：", annoation["answer"], "第{}个问题".format(total_question))
               total_question+=1
               if  is_contains_chinese(annoation["question"]):
                   pic_list.append(train["image"][index])
            index+=1
        print("一共有{a}张图片".format(a=index),"一个有{s}个问题".format(s=total_question))
        f.close()
    return  pic_list
#对列表进行去重并排序
# pic_list=get_chinese_pic_list_of_ques()
# new_list=list(set(pic_list))
# new_list.sort(key=pic_list.index)
# print(new_list)
# print("有{a}个中文问题".format(a=len(pic_list)))
# print("有{a}张图片用中文提问".format(a=len(new_list)))


#将指定名的图片复制到指定文件夹
def copy_pic_to_dir(pic_name):
    pic_dir='/data1/home/fzq/projects/mmf/data/datsets/EST-VQA-v1.0/images/test'
    dest_dir="/data1/home/fzq/projects/mmf/data/datsets/EST-VQA-chinese/images/test"
    file_list = os.listdir(pic_dir)
    for image in file_list:
        #如果图像名为B.png 则将B.png复制到F:\\Test\\TestA\\class
        if image == pic_name:
            if os.path.exists(os.path.join(pic_dir)):
                shutil.copy(os.path.join(pic_dir,image), os.path.join(dest_dir))
            else:
                os.makedirs(os.path.join(dest_dir))
                shutil.copy(os.path.join(pic_dir, image), os.path.join(dest_dir))
#将所有中文问题的图片返稿一个文件夹
pic_list=get_chinese_pic_list_of_ques()
new_list=list(set(pic_list))
new_list.sort(key=pic_list.index)
for pic in pic_list:
    copy_pic_to_dir(pic)