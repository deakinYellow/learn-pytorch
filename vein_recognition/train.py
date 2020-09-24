#-*-coding: utf-8-*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import  torch.nn.functional as f

from model import VeinRecognitionNetModel

plt.ion()   # interactive mode

##加载数据
#　训练需要做数据增强和数据标准化
#    验证只需要做数据标准化
#    很重要
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip( ),       
        transforms.ToTensor( ),       
        transforms.Normalize([0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )       ##图像标准化
    ]),
    'val': transforms.Compose([
        # transforms.Resize( (320,128)),
        # transforms.CenterCrop(224 ),       
        transforms.ToTensor( ),       
        transforms.Normalize([0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )       
    ]),
}

# ml_data_path='/home/deakin/ml/data/'
# data_dir =  ml_data_path + 'hymenoptera_data'
# ml_data_path='/home/deakin/Desktop/fingers/'
# data_dir =  ml_data_path + 'sdu01'
ml_data_path  ="/media/deakin/00000354000386C6/work/finger-vein/finger-17-50"
data_dir =  ml_data_path

image_datasets = { 
    x: datasets.ImageFolder( os.path.join( data_dir , x ), data_transforms[ x ] )
   for x in ['train', 'val' ]  }

### batch_size 很重要太小可能会导致训练出现nan
dataloaders = {
    x: torch.utils.data.DataLoader( image_datasets[ x ], batch_size= 12, shuffle=True, num_workers= 4  )
    for x  in ['train', 'val' ]
}

dataset_sizes = { x: len( image_datasets[x]  ) for x in ['train', 'val' ]   }
print( "dataloader: ----", dataset_sizes  )
class_names = image_datasets['train'].classes
use_cuda = torch.cuda.is_available()


## 显示图片
def imshow(  inp, title=None  ):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose( ( 1 , 2, 0 ) )     ## 显示坐标轴相关？
    mean = np.array( [ 0.485, 0.456, 0.406 ] )   ##跟色彩有关
    std =  np.array( [ 0.229, 0.224, 0.225 ] )  ##跟色彩有关

    inp = std * inp + mean
    inp = np.clip( inp, 0, 1 )

    plt.imshow( inp )

    if title is not None:
        plt.title( title )
    plt.pause( 0.1 )   #暂停一会，让 plots 更新

## 显示模型预测效果
# 写一个处理少量图片，并显示预测效果的通用函数
def  visualize_model(  model ,  dataloaders,  num_images , use_cuda ):

    images_so_far = 0
    fig = plt.figure(  figsize=(9,6) )

    for i, data in  enumerate( dataloaders['val'] ):
        inputs,  labels = data
        if use_cuda:
            inputs = Variable(  inputs.cuda( ) )
            labels = Variable(  labels.cuda( ) )
            print( "labels: {}".format( labels ) )
        else:
            inputs, labels = Variable( inputs ), Variable( labels )

        ##设置模型为预测模式,很重要，否侧测试单张图像会出错
        with torch.no_grad():
            model.eval()

        outputs = model(  inputs  )
        outputs = outputs / 1000   ##需要缩放，否则　softmax计算结果基本都是0 和１
        # print("outputs: {}".format(  outputs  ) )
        ##取softmax值,可代表概率
        softmax = f.softmax( outputs , dim = 1  )
        # print( "softmax: ", softmax )
        scores, preds = torch.max( softmax, 1 )
        print( "scores: ", scores )

        ##打印预测结果
        # print( "preds: ", preds )
        time.sleep( 1 )

        for j in range( inputs.size( )[ 0 ]  ):
            images_so_far += 1
            ax = plt.subplot(  num_images // 3,  3,  images_so_far )
            ax.axis( 'off' )
            if(  labels.data[ j ] == preds.data[ j ]  ):
                ret = "true"
            else:
                ret = "false"
            
            now_score =  round( scores[ j ].item( ) , 2  ) 
            probability = str( now_score  )
            ax.set_title(  't: {} '.format( class_names[ labels[ j ] ] ) + ' p: {} '.format( class_names[  preds[ j ]  ]  )  +"score: "  +  probability + " ret: " + ret  )
            imshow(  inputs.cpu( ).data[ j ]  )

            if images_so_far == num_images:
                return

if __name__ == "__main__":
    # model_conv  = train_model(  model_conv , criterion, optimizer_conv , exp_lr_scheduler, num_enpoches=20   )
    # visualize_model( model_conv , 4   )
    # time.sleep( 30 )
    print("===" , dataloaders )
    pre_model_params_file = "vein_model_params.pkl"

    print("==============Vein Recognition Train Test========================")
    ##生成模型对象
    print( "cuda is: ", use_cuda )
    vein_net = VeinRecognitionNetModel( 10 , 0.001, 50,  use_cuda  )

    ##从磁盘加载历史模型
    vein_net.load_model_params(  pre_model_params_file  )

    ##模型训练
    # ret_model = vein_net.train( dataloaders, dataset_sizes, 500, use_cuda )
    ###保存模型到磁盘 , 仅保存和加载模型参数(推荐使用)
    # torch.save( ret_model.state_dict(),  pre_model_params_file  )

    ###测试效果
    visualize_model(  vein_net.model ,  dataloaders, 24,  use_cuda )
    input(">>>")

