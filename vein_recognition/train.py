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

from model import VeinRecognitionNetModel

plt.ion()   # interactive mode

##加载数据
#　训练需要做数据增强和数据标准化
#    验证只需要做数据标准化

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop( 224 ),
        transforms.RandomHorizontalFlip( ),       
        transforms.ToTensor( ),       
        transforms.Normalize([0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )       
    ]),
    'val': transforms.Compose([
        transforms.Resize( 224 ),
        transforms.CenterCrop(224 ),       
        transforms.ToTensor( ),       
        transforms.Normalize([0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )       
    ]),
}

# ml_data_path='/home/deakin/ml/data/'
# data_dir =  ml_data_path + 'hymenoptera_data'

ml_data_path='/home/deakin/Desktop/fingers/'
data_dir =  ml_data_path + 'sdu01'

image_datasets = { 
    x: datasets.ImageFolder( os.path.join( data_dir , x ), data_transforms[ x ] )
   for x in ['train', 'val' ]  }

dataloaders = {
    x: torch.utils.data.DataLoader( image_datasets[ x ], batch_size= 6, shuffle=True, num_workers= 4  )
    for x  in ['train', 'val' ]
}
dataset_sizes = { x: len( image_datasets[x]  ) for x in ['train', 'val' ]   }
print( "dataloader: ----", dataset_sizes  )
class_names = image_datasets['train'].classes
use_cuda = torch.cuda.is_available()



## 显示图片
def imshow( inp, title=None ):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose( ( 1, 2, 0 ) )
    mean = np.array( [ 0.485, 0.456, 0.406 ] )
    std =  np.array( [ 0.229, 0.224, 0.225 ] )

    inp = std * inp + mean
    inp = np.clip( inp, 0, 1 )

    plt.imshow( inp )

    if title is not None:
        plt.title( title )
    plt.pause( 0.1 )   #暂停一会，让 plots 更新

## 显示模型预测效果
# 写一个处理少量图片，并显示预测效果的通用函数
def  visualize_model( model, num_images=6 ):
    images_so_far = 0
    fig = plt.figure( )
    for i, data in  enumerate( dataloaders['val'] ):
        inputs,  labels = data
        if use_gpu:
            inputs = Variable(  inputs.cuda( ) )
            labels = Variable(  labels.cuda( ) )
        else:
            inputs, labels = Variable( inputs ), Variable( labels )

        outputs = model(  inputs  )
        _, preds = torch.max( outputs.data, 1 )

        for j in range( inputs.size( )[ 0 ] ):
            images_so_far += 1
            ax = plt.subplot( num_images // 2 ,  2,  images_so_far )
            ax.axis( 'off' )
            ax.set_title( 'predicted: {}'.format( class_names[ preds[ j ] ]  ) )
            imshow(  inputs.cpu( ).data[ j ]  )

            if images_so_far == num_images:
                return


## 显示图片
def imshow( inp, title=None ):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose( ( 1, 2, 0 ) )
    mean = np.array( [ 0.485, 0.456, 0.406 ] )
    std =  np.array( [ 0.229, 0.224, 0.225 ] )

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
    fig = plt.figure(  figsize=(9,6))
    for i, data in  enumerate( dataloaders['val'] ):
        inputs,  labels = data
        if use_cuda:
            inputs = Variable(  inputs.cuda( ) )
            labels = Variable(  labels.cuda( ) )
        else:
            inputs, labels = Variable( inputs ), Variable( labels )

        outputs = model(  inputs  )
        _, preds = torch.max( outputs.data, 1 )

        ##打印预测结果
        print( "preds: ", preds )

        time.sleep( 1 )

        for j in range( inputs.size( )[ 0 ]  ):
            images_so_far += 1
            ax = plt.subplot(  num_images // 2 ,  2,  images_so_far )
            ax.axis( 'off' )
            if(  labels.data[ j ] == preds.data[ j ]  ):
                ret = "true"
            else:
                ret = "false"
            ax.set_title(  'truth: {} '.format( class_names[ labels[ j ] ] ) + ' predicted: {} '.format( class_names[  preds[ j ]  ]  )  + "ret: " + ret  )
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
    vein_net = VeinRecognitionNetModel( 6 , 0.0001, use_cuda  )
    ##从磁盘加载历史模型
    vein_net.load_model_params(  pre_model_params_file  )

    ##模型训练
    ret_model = vein_net.train( dataloaders, dataset_sizes, 300, use_cuda )
    ###保存模型到磁盘 , 仅保存和加载模型参数(推荐使用)
    ##torch.save( vein_model.state_dict(),  pre_model_params_file  )

    visualize_model(  ret_model ,  dataloaders,  12  , use_cuda )

    input(">>>")











