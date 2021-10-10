"""
采用Robert  对text进行enbedding    大小用xcodelength表示   预训练model后面可以更改
https://blog.csdn.net/weixin_30034903/article/details/113399809?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Eessearch%7Evector-11.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Eessearch%7Evector-11.no_search_link
"""

import torch
# from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from torch import nn
from transformers import  BertModel, BertConfig,BertTokenizer

# 自己下载模型相关的文件，并指定路径
config_path = 'D:\\新建文件夹\\学习\\预训练模型\\RoBERTa_zh_L12_PyTorch\\config.json'
model_path = 'D:\新建文件夹\学习\预训练模型\RoBERTa_zh_L12_PyTorch\\pytorch_model.bin'
vocab_path = 'D:\新建文件夹\学习\预训练模型\RoBERTa_zh_L12_PyTorch\\vocab.txt'


# ——————构造模型——————
class BertTextNet(nn.Module):
    def __init__(self, code_length):
        super(BertTextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(
            model_path, config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size


        #直接输出原始维数
        self.fc = nn.Linear(embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features=text_embeddings
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


textNet = BertTextNet(code_length=2048)

# ——————输入处理——————
tokenizer = BertTokenizer.from_pretrained(vocab_path)

texts = ["[CLS] 国庆节期间天气都很好，适合野炊 [SEP]",
         "[CLS] 希望大家天天开心，心想事成 [SEP]"]
tokens, segments, input_masks = [], [], []
for text in texts:
    tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
    tokens.append(indexed_tokens)
    segments.append([0] * len(indexed_tokens))
    input_masks.append([1] * len(indexed_tokens))

max_len = max([len(single) for single in tokens])  # 最大的句子长度

for j in range(len(tokens)):
    padding = [0] * (max_len - len(tokens[j]))
    tokens[j] += padding
    segments[j] += padding
    input_masks[j] += padding
# segments列表全0，因为只有一个句子1，没有句子2
# input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
# 相当于告诉BertModel不要利用后面0的部分

# 转换成PyTorch tensors
tokens_tensor = torch.tensor(tokens)
segments_tensors = torch.tensor(segments)
input_masks_tensors = torch.tensor(input_masks)

# ——————提取文本特征——————
text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
print("转换的句子为：",texts)
print(text_hashCodes)
print(text_hashCodes.size())

