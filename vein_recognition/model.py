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

""" 
静脉分类网络模型
"""
class VeinRecognitionNetModel:

    def __init__( self , classes_size ,  learnning_rate, use_cuda ):
        ## ----------------------模型创建-----------------------------------
        self.classes_size  = classes_size

        self.model = torchvision.models.resnet50( pretrained=True )    #resnet50 预训练模型
        ####禁止更新所有网络参数
        for param in self.model.parameters( ):
            param.requires_grad = False             
        # 新构建的　module 的参数中，默认设置 requires_grad=True
        num_ftrs = self.model.fc.in_features
        self.model.fc  = nn.Linear( num_ftrs,  2  )

        if use_cuda:
            self.model  = self.model.cuda( )

         # 损失函数选取, 交叉熵做损失
        self.criterion = nn.CrossEntropyLoss( )

        # 只对最后一层的参数进行优化
        # 选择动量法对参数进行优化
        self.optimizer = optim.SGD(  self.model.parameters( ), lr = learnning_rate , momentum= 0.9 )
        # 每７轮迭代学习率变为原来的0.1
        self.scheduler = lr_scheduler.StepLR( self.optimizer , step_size = 7 ,  gamma= 0.1  )

    ###　参数： #  数据 # 迭代次数
    def  train( self ,  dataloaders ,  dataset_sizes,  num_enpoches = 25, use_cuda = True  ):
        since = time.time()
        #用于保存最好的模型参数
        best_model_wts = copy.deepcopy(  self.model.state_dict( ) )
        best_acc = 0.0
        ###进入迭代
        for epoch in range( num_enpoches ):
            print('Epoch {} / {}'.format(  epoch, num_enpoches - 1 )  )
            print( '-' * 10 )
            # 每一次迭代都有训练和验证阶段
            for phase in ['train', 'val' ]:
                if phase == 'train':
                    self.model.train( True )   # 设置 model 为训练模式
                else:
                    self.model.train( False ) # 设置 model 为评估( evaluate ) 模式,不更新参数

                running_loss = 0.0
                running_corrects = 0
                # 遍历数据
                for data in dataloaders[ phase ]:
                    # 获取输入
                    inputs, labels = data

                    #　用 Variable 包装输入数据
                    if use_cuda:
                        inputs = Variable(  inputs.cuda( )  )
                        labels = Variable(  labels.cuda( )  )
                    else:
                        inputs, labels = Variable( inputs ),  Variable( labels )

                    # 设置梯度参数为０
                    self.optimizer.zero_grad( )

                    print("================")
                    print('===========inputs size: {:d} '.format( inputs.__len__( )  ) ) 
                    #正向传递
                    outputs = self.model(  inputs  )

                    _, preds =  torch.max( outputs.data, 1 )
                    #计算损失
                    loss = self.criterion( outputs, labels )

                    #如果是训练阶段，向后传递和优化
                    if phase == 'train':
                        loss.backward( )
                        self.optimizer.step( )
                        #print("update params.................")

                    # 统计训练过程参数
                    running_loss += loss.data.item() * inputs.size( 0 )
                    running_corrects +=  torch.sum( preds == labels.data )

                    epoch_loss = running_loss  /  dataset_sizes[ phase ]
                    epoch_acc =  running_corrects.double()  /  dataset_sizes[ phase ]

                    print('{} Loss: {:.4f} Acc: {:.4f} '.format(  phase,  epoch_loss, epoch_acc  )  )

                    #如果是训练阶段，优化学习率
                    if phase ==  'train':
                        ##学习率优化
                        self.scheduler.step()

                # 深拷贝model, 记录最好的模型参数
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(  self.model.state_dict( )  )
            print( '-' * 20 )

        ###统计训练时长
        time_elapsed = time.time( ) - since
        print("Trainning complete in {:.0f}  m {:.0f}  s  ".format( time_elapsed  // 60, time_elapsed  %  60 ) )
        print('Best val Acc: {:.4f}'.format(  best_acc ) )
        #加载最佳的权重参数,并返回模型
        self.model.load_state_dict(  best_model_wts   )
        return self.model


if __name__ == "__main__":
    pass

