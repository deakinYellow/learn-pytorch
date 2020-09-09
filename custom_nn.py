
#-*-coding: utf-8-*-

import torch
from torch.autograd import Variable


class TwoLayerNet( torch.nn.Module ):
    def __init__( self, D_in, H, D_out ):
        """
        在构造函数中，我们实例化两个　nn.Linear 模块并将它们分配为成员变量
        """
        super( TwoLayerNet, self ).__init__()

        self.linear1 = torch.nn.Linear( D_in, H )
        self.linear2 = torch.nn.Linear( H,  D_out  )

    def forward(self, x ):
        """
        在　forward 函数中，我们接收一个变量的数据输入，我们必须返回一个变量的输出数据
        我们可以使用构造函数中的定义的模块以及变量上的任意运算符
        """
        h_relu = self.linear1( x ).clamp( min = 0 )
        y_pred = self.linear2( h_relu )
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
model = TwoLayerNet( D_in, H, D_out )



# nn 还包含了流行的损失函数的定义
# 在这种情况下，我们将使用均方差（ＭSE）作为我们的损失函数
loss_fn = torch.nn.MSELoss( size_average = False )

# 使用优化包来定义一个优化器，它将为我们更新模型的权重
# 在这里，我们将使用 Adam;  这个 optim 包含许多其他优化算法
# Adam构造函数的第一个参数告诉优化器应该更新哪个　Variables.

learning_rate = 1e-4
optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate )

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

















