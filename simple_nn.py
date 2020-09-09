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
x = Variable( torch.randn(  N, D_in).type( dtype  )  )
y = Variable( torch.randn(  N, D_out).type( dtype  ) , requires_grad=False )


## 使用 nn 包将我们的模型定义为一些列图层
## nn.Sequential 是一个包含其他模块的模块，并将它们按顺序应用以产生其输出
# 每个线性模块使用线性函数计算来自输入的输出，并保存内部变量的权重和偏差

model = torch.nn.Sequential(
    torch.nn.Linear( D_in, H ),
    torch.nn.ReLU(),
    torch.nn.Linear( H, D_out )
)

# nn 还包含了流行的损失函数的定义
# 在这种情况下，我们将使用均方差（ＭSE）作为我们的损失函数
loss_fn = torch.nn.MSELoss( size_average = False )


learning_rate = 1e-4
for t in range(100):    
    # 正向传递，通过x传递给模型来计算预测的y
    # 模块对象会覆盖__call__ 运算符，
    y_pred = model( x )
    #计算和打印损失
    # 我们传递包含y的预测值和真实值的变量，并且损失函数返回损失的变量
    loss = loss_fn( y_pred, y )
    print(  t,':',  loss.data.item()  )

    # 在运行反向传递前将梯度归零
    model.zero_grad()

    # 后向传递：计算相对于模型的所以可以学习参数的损失梯度
    # 在内部，每个模块的参数都存储在变量　require_grad = True 中，
    # 因此该调用计算模型中所有可学习参数的梯度
    loss.backward( )

    # 更新权重
    # 每个参数都是一个变量，所以我们可以像之前那样访问它的数据和梯度
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data






