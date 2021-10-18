
#生成答案的静态词典   按字进行分词
import pandas
import json
import  re

#检验是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

#只保留中文字符
def remove(text):
    remove_chars = r'[^\u4e00-\u9fa5]'
    return re.sub(remove_chars, '', text)

with open("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\annotations\\train.json")  as f:
    train = json.load(f, encoding="utf8")
    answers_list=[]
    train = pandas.DataFrame(data=train)
    pandas.set_option('display.max_columns', None)
    for i in train["annotation"]:
        for annoation in i:
            #仅处理中文部分
            if is_contains_chinese(annoation["question"]):
                answers_list.append(annoation["answer"])
    print(answers_list,len(answers_list))
