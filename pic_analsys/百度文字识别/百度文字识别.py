import glob
from os import path
import os
from aip import AipOcr
from PIL import Image
"""百度最基本的ocr接口 效果差"""
'''使用时只需要修改百度智能云接口，位于baiduOCR函数，和图片存放位置，位于倒数第三行'''

def baiduOCR(outfile):
    """利用百度api识别文本，并保存提取的文字
    picfile:    图片文件名
    outfile:    输出文件
    """
    filename = path.basename(picfile)

    APP_ID = '24803470'
    API_KEY = '9GsUi84NnbYhcelpZKw2sYp0'
    SECRET_KEY = 'vQ1211ty85Ip13KdhctpHK8qcnnDrsky'
    client = AipOcr(APP_ID, API_KEY,SECRET_KEY)

    i = open(picfile, 'rb')
    img = i.read()
    print("正在识别图片：\t" + filename)
    message = client.basicGeneral(img)  # 通用文字识别，每天 500 次免费
    # message = client.basicAccurate(img)   # 通用文字高精度识别，每天 500 次免费
    print("识别成功！")
    i.close()

    with open(outfile, 'a+',encoding='utf-8') as fo:
        fo.writelines("+" * 60 + '\n')
        fo.writelines("识别图片：\t" + filename + "\n" * 2)
        fo.writelines("文本内容：\n")
        # 输出文本内容
        for text in message.get('words_result'):
            fo.writelines(text.get('words') + '\n')
        fo.writelines('\n' * 2)
    print("文本导出成功！")
    print()

if __name__ == "__main__":
    open('result.txt', 'a+', encoding='utf-8').close()
    outfile = 'result.txt'
    for picfile in glob.glob("D:\\Download\\sample_picture\\pictures\\*"):
        baiduOCR(outfile)
    print('图片文本提取结束！文本输出结果位于 %s 文件中。' % outfile)

