# coding=utf-8

import torch
from  torch import Tensor
import   numpy as np

#-----------------------tensor  create--------------------------------------

###直接创建
n = 2
tens  = torch.eye( n  )     #返回２维张量，对角为１其余为０
print( tens )
if n == 2:
    print( torch.is_tensor(  tens ) )   ##判断是否为张量对象

###从numpy创建
#Numpy桥，将numpy.ndarray 转换为pytorch的 Tensor。
#返回的张量tensor和numpy的ndarray共享同一内存空间。
#修改一个会导致另外一个也被修改。返回的张量不能改变大小。
a = np.array([1,2,3,])
t = torch.from_numpy( a )
print( t )
t[ 0 ] = -1
print( a )

##torch.linspace
#torch.linspace( start, end, steps, out=None )  返回一个１维张量，包含区间[ start, end ]上均匀间隔的steps个点。张量长度为steps
tl = torch.linspace( 0, 10 ,11  )
print('torch linespace: \n ',  tl )

###返回数值全为１的张量
tone = torch.ones( 2, 3 )
print( "torch one: \n " , tone )


# .................












