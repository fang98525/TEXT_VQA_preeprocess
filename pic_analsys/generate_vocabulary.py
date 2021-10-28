
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

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


with open("D:\\新建文件夹\\数据集处理\\EST-VQA-v1.0\\annotations\\train.json")  as f:
    train = json.load(f, encoding="utf8")
    answers_list=[]
    vocabulary=[]
    train = pandas.DataFrame(data=train)
    pandas.set_option('display.max_columns', None)
    for i in train["annotation"]:
        for annoation in i:
            #仅处理中文部分
            if is_contains_chinese(annoation["question"]):
                answers_list.append(annoation["answer"])
                break


    #
    for answera in answers_list:
        answera = answera.replace("·", " ")
        answera=answera.replace(","," ")
        answera=answera.replace("、", " ")
        answera=answera.replace("，", "")
        answera=str(answera).replace("!"," ")
        if answera.find(" "):
            answera=answera.split(" ")
        for answer in answera:


            if is_all_chinese(answer):
                for token in answer:
                    vocabulary.append(token)
            elif str(answer).isdigit():
                vocabulary.append(answer)
            else:
                answer=answer.split("\n")
                for i in answer:
                    vocabulary.append(i)
    vocabulary=list(set(vocabulary))
    print(len(vocabulary))  #4636

    with open("EST_answer_vocab.txt", "w", encoding="utf8") as f:
        f.writelines([w + "\n" for w in vocabulary])

    # print(answers_list,len(answers_list[:8000]))
    # with open("EST_answer_vocab_my_deal.txt","r",encoding="utf8") as f:
    #     data=f.readlines()
    #     print(data)

