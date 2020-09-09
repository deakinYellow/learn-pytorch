#-*-coding: utf-8-*-

import random
import torch
from torch.autograd import Variable

"""
为了展示Pytorch的动态图的强大，我们实现了一个非常奇异的模型：一个全连接的ReLU激活的神经网络，
每次前向计算时都随机选择一个１到４之间的数字n，然后接下来就有n层隐藏层,每个隐藏层的连接权重共享。
"""

class DynamicNet( torch.nn.Module ):
    def __init__( self, D_in, H, D_out ):
        """
        在构造函数中，我们实例化3个　nn.Linear 我们将在正向传递中使用它们
        """
        super( DynamicNet, self ).__init__()

        self.input_linear = torch.nn.Linear( D_in, H )

        self.middle_linear = torch.nn.Linear( H, H )

        self.output_linear = torch.nn.Linear( H,  D_out  )

    def forward(self, x ):
        """
        对于模型的正向通道，我们随机选择０，１，２，３
        并重复使用多次计算隐藏层表示的 middle_linear 模块
        由于每个正向通道都会生成一个动态计算图，因此在定义模型的正向通道时，
        我们可以使用普通的python控制流操作符（如循环或条件语句）

        在这里我们看到，定义计算图时多次重复操作相同模块是完全安全的，
        这是　Lua Torch 的一大改进
        """
        h_relu = self.input_linear(  x ).clamp( min = 0 )

        # 　n 个隐藏层
        for _ in range( random.randint( 0, 3 )  ):
            h_relu = self.middle_linear( h_relu ).clamp( min = 0 )

        y_pred = self.output_linear( h_relu  )
        return y_pred


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

## 使用 nn 包自定义图层
model = DynamicNet( D_in, H, D_out )

# nn 还包含了流行的损失函数的定义
# 在这种情况下，我们将使用均方差（ＭSE）作为我们的损失函数
loss_fn = torch.nn.MSELoss( size_average = False )

# 使用优化包来定义一个优化器，它将为我们更新模型的权重
# 用随机下降训练这个奇怪的模型非常困难，所以我们使用动量
learning_rate = 1e-4

#随机下降法
# optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate  )
#动量法
optimizer = torch.optim.SGD( model.parameters(), lr=learning_rate, momentum = 0.9  )

for t in range(500):    
    # 正向传递，通过x传递给模型来计算预测的y
    # 模块对象会覆盖__call__ 运算符，
    y_pred = model( x )
    #计算和打印损失
    # 我们传递包含y的预测值和真实值的变量，并且损失函数返回损失的变量
    loss = loss_fn( y_pred, y )
    print(  t,':',  loss.data.item()  )

    # 在后向传递之前，使用优化器对象为其要更新的变量（这是模型的可学习的权重） 的所以梯度归零。
    # 这是因为默认情况下，只要调用.backward(),梯度就会在缓存区中累积（即不会被覆盖）
    # 查看 torch.autograd.backward 的文档获取详情
    optimizer.zero_grad()
    # 后向传递：计算相对于模型的所以可以学习参数的损失梯度
    loss.backward( )
    #在优化器上调用 step 函数会更新其参数
    optimizer.step()
