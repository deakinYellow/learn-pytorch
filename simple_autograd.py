#-*-coding: utf-8-*-

import torch
from torch.autograd import Variable

dtype  =  torch.FloatTensor
# dtype = torch.cuda.FloatTensor   #GPU上运行


# N 是一个batch的样本数量; D_in是输入维度;
# H 是隐藏层向量的维度; D_out是输出维度.
N, D_in, H, D_out = 64, 1000, 100, 10


#  创建随机的输入输出数据,并将它们保存在变量中
#  标准正态分布中随机选取
#  设置　requires_grad = False, 因为在后向传播时，我们并不需要计算关于这些变量梯度 
x = Variable( torch.randn(  N, D_in).type( dtype  ) , requires_grad=False )
y = Variable( torch.randn(  N, D_out).type( dtype  ) , requires_grad=False )

# 随机初始化权重参数,在计算后向传播时需要计算梯度
w1 =  Variable( torch.randn(  D_in, H ).type( dtype  ) , requires_grad=True )
w2 =  Variable( torch.randn(  H, D_out ).type( dtype  ) , requires_grad=True )


learning_rate = 1e-6
for t in range(500):    
    # 前向计算, 算出y的预测值
    y_pred = x.mm( w1 ).clamp( min=0 ).mm( w2 )

    #使用变量上的操作计算和打印损失
    #现在损失是形状变量（１）并且　loss.data 是形状的张量　；　loss.data[ 0 ] 是持有损失的标量值
    loss = ( y_pred - y  ).pow( 2 ).sum()
    #print( t, loss.item() )
    print(  t,':',  loss.data.item()  )

    # 使用autograd来计算反向传递
    # 该调用将使用　requires_grad = True 来计算相对于所有变量的损失梯度
    # 在这次调用之后　w1.grad 和　w2.grad 将是变量
    # 它们分别相对于w1 和　w2 保存损失的梯度
    loss.backward( )

    # 使用梯度下降更新权重；　w1.data 和　w2.data 是张量
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 更新权重后手动将梯度重新归零
    w1.grad.data.zero_()
    w2.grad.data.zero_()


