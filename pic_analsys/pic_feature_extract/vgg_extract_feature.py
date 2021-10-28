import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

data_dir = 'D:\\新建文件夹\\数据集处理\\examples_pic'  # ocr_lmdb_file
features_dir = 'D:\\新建文件夹\\数据集处理\\feature'  # Vgg_features_train


""""
提取dir下所有图片的feature       目前使用这个，后面在测试效果   使用cpu  gou 用不了

"""

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.linear=nn.Linear(4096,2048)

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        output=self.linear(output)  # 修改输出为2048维
        return output

torch.cuda.empty_cache()
model = Encoder()
# model = model.cuda()

#直接提取区域
def extractor_imgcode(img_code,net,  use_gpu=False):


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )


    img = transform(img_code)
    print("box的尺寸为：",img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    # print("输出特征size：",y.shape)
    # print("当前box的feature：",y)
    # np.savetxt(saved_path, y, delimiter=',')
    return y



def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)
    # print("图片的尺寸为：",img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    print("输出特征size：",y.shape)
    print(y[:100])

    # np.savetxt(saved_path, y, delimiter=',')


if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    files_list = []
    x = os.walk(data_dir)
    for path, d, filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))

    print(files_list)

    use_gpu = torch.cuda.is_available()
    use_gpu=False

    for x_path in files_list:
        print("图片" + x_path)
        file_name = x_path.split('/')[-1]
        fx_path = os.path.join(features_dir, file_name + '.txt')
        print("保存路径为：",fx_path)
        extractor(x_path, fx_path, model, use_gpu)