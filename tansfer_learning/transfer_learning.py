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

plt.ion()   # interactive mode

print("transfer learning.")

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

print( "----", dataset_sizes  )

class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()


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

# 训练模型
def train_model( model, criterion, optimizer, scheduler, num_enpoches=25 ):
    since = time.time()
    best_model_wts = copy.deepcopy(  model.state_dict( ) )
    best_acc = 0.0

    for epoch in range( num_enpoches ):

        print('Epoch {} / {}'.format(  epoch, num_enpoches - 1 )  )
        print( '-' * 10 )

        # 每一次迭代都有训练和验证阶段
        for phase in ['train', 'val' ]:

            if phase == 'train':
                model.train( True )   # 设置 model 为训练模式
            else:
                model.train( False ) # 设置 model 为评估( evaluate ) 模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for data in dataloaders[ phase ]:
                # 获取输入
                inputs, labels = data
                #　用 Variable 包装输入数据
                if use_gpu:
                    inputs = Variable(  inputs.cuda( )  )
                    labels = Variable(  labels.cuda( )  )
                else:
                    inputs, labels = Variable( inputs ),  Variable( labels )

                # 设置梯度参数为０
                optimizer.zero_grad( )

                #正向传递
                outputs = model(  inputs  )
                _, preds = torch.max( outputs.data, 1 )
                loss = criterion( outputs, labels )

                #　如果是训练阶段，向后传递和优化
                if phase == 'train':
                    loss.backward( )
                    optimizer.step( )
                    # print("update params.................")

                # 统计
                running_loss += loss.data.item() * inputs.size( 0 )
                running_corrects += torch.sum( preds == labels.data )

            epoch_loss = running_loss  /  dataset_sizes[ phase ]
            epoch_acc =  running_corrects.double()  /  dataset_sizes[ phase ]

            print('{} Loss: {:.4f} Acc: {:.4f} '.format(  phase,  epoch_loss, epoch_acc  )  )

            if phase ==  'train':
                    pass
                    ##学习率优化
                    scheduler.step()

            # 深拷贝model, 记录最好的状态
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(  model.state_dict( )  )

        print( '-' * 20 )

    time_elapsed = time.time( ) - since
    print("Trainning complete in {:.0f}  m {:.0f}  s  ".format( time_elapsed  // 60, time_elapsed  %  60 ) )

    print('Best val Acc: {:.4f}'.format(  best_acc ) )

    #加载最佳模型的权重
    model.load_state_dict(  best_model_wts   )
    return model

## 显示模型预测效果
# 写一个处理少量图片，并显示预测效果的通用函数
def  visualize_model(  model, num_images=6  ):
    images_so_far = 0
    fig = plt.figure(  figsize=(9,6))
    for i, data in  enumerate( dataloaders['val'] ):
        inputs,  labels = data
        if use_gpu:
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

### 场景一 微调CNN,实现二分类
"""
这里我们使用resnet18作为我们的初始网络，在自己的数据集上继续训练预训练好的模型，
所不同的是，我们修改原网络最后的全连接层输出维度为2，因为我们只需要预测是蚂蚁还是蜜蜂，
原网络输出维度是1000，预测了1000个类别
"""
def  classification2():
    #  调整卷积神经网络
    #  加载一个预训练的网络，并重置最后一个全连接层。
    model_ft = models.resnet18(  pretrained=True )
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear( num_ftrs, 6  )

    if use_gpu:
        model_ft = model_ft.cuda( )

    # 如你所见，所有参数都将被优化
    # 使用分类交叉熵Cross-Entropy做损失函数
    criterion = nn.CrossEntropyLoss( )
    optimizer_ft = optim.SGD(  model_ft.parameters( ) , lr = 0.001,  momentum =  0.9   )
    # optimizer_ft = optim.Adam( model_ft.parameters( ), lr = 0.001  )
    # 每７个迭代让 learning_rate  衰减0.1 因素
    exp_lr_scheduler = lr_scheduler.StepLR( optimizer_ft, step_size = 20 ,  gamma= 0.1  )
    ## 训练和评估。如果使用CPU 将花费　15-25分钟，使用GPU 将少于１分钟
    model_ft  = train_model(  model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_enpoches=300  )
    visualize_model( model_ft )
    input(">>>")

##  场景二 , 卷积神经网络作为固定的特征提取器
"""
    这里，我们固定网络中除最后一层外的层的所有权重。为固定这些参数，我们需要设置　
    requires_grad = False , 然后在　backward( ) 中就不会计算梯度
"""
def  feature_extraction():

    model_conv = torchvision.models.resnet50( pretrained=True )
    for param in model_conv.parameters( ):
        param.requires_grad = False             

    # 新构建的　module 的参数中，默认设置 requires_grad=True
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear( num_ftrs, 6 )

    if use_gpu:
        model_conv = model_conv.cuda( )

    # 损失函数选取
    criterion = nn.CrossEntropyLoss( )

    # 只对最后一层的参数进行优化
    optimizer_conv = optim.SGD( model_conv.fc.parameters( ), lr = 0.001, momentum= 0.9 )
    exp_lr_scheduler = lr_scheduler.StepLR( optimizer_conv , step_size = 100 ,  gamma= 0.1  )
    ## 训练和评估。如果使用CPU 将花费　15-25分钟，使用GPU 将少于１分钟
    model_conv  = train_model(  model_conv , criterion, optimizer_conv , exp_lr_scheduler, num_enpoches=200   )

    visualize_model( model_conv ,  12  )
    ### 等待结束
    input(">>>")

if __name__ == "__main__":
    #classification2()
    feature_extraction( )
    """
    while True:
        # 获取一批训练数据
        inputs, classes = next( iter(dataloaders['train' ] ) )
        # 从这批数据生成一个方格
        out = torchvision.utils.make_grid( inputs )
        imshow( out , title = [ class_names[x] for x in classes ] )
        time.sleep( 1 )
     """

