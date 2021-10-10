"""
fasttext 获得词嵌入   用服务器
"""
import fasttext
import fasttext.util
from torch import nn
import torch

class NET2VEC(nn.Module):

    def __init__(self,last_length):
        super(NET2VEC, self).__init__()
        self.fc=nn.Linear(300,last_length)


    def forward(self,input):
        output=self.fc(input)
        return output

# ftmodel=fasttext.load_model("/data1/home/fzq/projects/mmf/data/model/fasttext_word_model/wiki_ch/wiki.zh.bin")
# print(ftmodel.get_dimension())
# word_vector=ftmodel.get_sentence_vector("今天天气不错！")
# # print(ftmodel.words)
# print(word_vector)
# model=NET2VEC(last_length=768)
# vec_last=model(torch.from_numpy(word_vector))
# print(vec_last)
# print(vec_last.size())



#封装成 api
def change_word_vec_dim(text_list,last_length):
    """

    :param text:  输入的文本list   对每一句话转换成vector      原始为300 维
    :param last_length:  目标维数
    :return:    返回vec的目标维数
    """
    #加载模型
    ftmodel = fasttext.load_model("/data1/home/fzq/projects/mmf/data/model/fasttext_word_model/wiki_ch/wiki.zh.bin")
    # print(ftmodel.labels)
    vec_list=[]
    for text in  text_list:
        word_vector = ftmodel.get_sentence_vector(text)
        model = NET2VEC(last_length=last_length)
        vec_last = model(torch.from_numpy(word_vector))  # 目标vec
        vec_list.append(vec_last.detach().numpy())
    return vec_list

text=["我感觉很开心今天","如果时光能够倒流"]
encodding_dim=768
output=change_word_vec_dim(text_list=text,last_length=encodding_dim)
print(output[0])
print(output[0].size())
