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

    def __init__( self , classes_size ,  learnning_rate, learnning_rate_step,  use_cuda ):
        ## ----------------------模型创建-----------------------------------
        self.classes_size  = classes_size
        self.learnning_rate  = classes_size

        # self.model = torchvision.models.resnet50( pretrained=True )    #resnet50 预训练模型
        self.model = torchvision.models.resnet18( pretrained=True )
        
        ####禁止更新部分网络参数
        for i , param in enumerate( self.model.parameters( ) ):
            print("param: " , i  )
            if( i < 61 ) :
                param.requires_grad = False             

        ###重置最后全连接层数
        num_ftrs = self.model.fc.in_features
        self.model.fc  = nn.Linear( num_ftrs,  classes_size  )

        if use_cuda:
            self.model  = self.model.cuda( )

         # 损失函数选取, 交叉熵做损失
        self.criterion = nn.CrossEntropyLoss( )
        # self.criterion = nn.NLLLoss( )

        # 只对最后一层的参数进行优化
        # 选择动量法对参数进行优化
        # self.optimizer = optim.SGD(  self.model.parameters( ), lr =  self.learnning_rate , momentum= 0.9 )
        # 随机梯度下降法进行优化
        self.optimizer = optim.Adam( self.model.parameters(), lr=self.learnning_rate  )

        # 每n轮迭代学习率变为原来的0.5
        self.scheduler = lr_scheduler.StepLR( self.optimizer , learnning_rate_step ,  gamma= 0.5  )

    ### 从磁盘加载历史模型
    def load_model_params( self,  model_params_file_path ):
        self.model.load_state_dict(  torch.load( model_params_file_path ) )

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

                    # print('===========inputs size: {:d} '.format( inputs.__len__( )  ) ) 
                    #正向传递
                    # print("intputs: ", inputs[ 0 ] )
                    outputs = self.model(  inputs  )
                    #查看输出，输出范围没限制,最后一层为线性层
                    _, preds =  torch.max( outputs.data, 1 )
                    # print("outputs: ", outputs )
                    # print("preds: ", preds )
                    # print("labels: ", labels )
                    # labels:  tensor([0, 7, 6, 7], device='cuda:0')
                    #这里计算损失,每个预测的损失,所以必须包含outputs信息
                    loss = self.criterion( outputs, labels )

                    #如果是训练阶段，向后传递和优化
                    if phase == 'train':
                        loss.backward( )
                        self.optimizer.step( )
                        #print("update params.................")

                    # 统计
                    running_loss += loss.data.item() * inputs.size( 0 )
                    running_corrects +=  torch.sum( preds == labels.data )

                epoch_loss = running_loss  /  dataset_sizes[ phase ]
                epoch_acc =  running_corrects.double()  /  dataset_sizes[ phase ]

                print('{} Loss: {:.4f} Acc: {:.4f} '.format(  phase,  epoch_loss, epoch_acc  )  )

                #如果是训练阶段，优化学习率
                if phase ==  'train':
                    self.scheduler.step()

            # 深拷贝model, 记录验证效果最好的模型参数
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



