# -*- coding: utf-8 -*-
import  numpy as np

# N 是一个batch的样本数量; D_in是输入维度;
# H 是隐藏层向量的维度; D_out是输出维度.
#N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 2, 1000, 100, 10

# 创建随机的输入输出数据
##标准正态分布中随机选取
x = np.random.randn(N, D_in)     
print( x )
y = np.random.randn(N, D_out)

# 随机初始化权重参数
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)


learning_rate = 1e-6
for t in range(500):    
    # 前向计算, 算出y的预测值
    h = x.dot(w1)
    ###逐位比较选取最大值,这里取大于０, 作激活函数
    h_relu = np.maximum(h, 0)     
    ##print( h_relu  )
    y_pred = h_relu.dot(w2)

    # 计算并打印误差值
    loss = np.square(y_pred - y).sum()
    print(t)

    # 在反向传播中, 计算出误差关于w1和w2的导数
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

 ###测试模型效果
x_test = np.random.randn(N, D_in)     
  # 前向计算, 算出y的预测值
h = x_test.dot(w1)
 ###逐位比较选取最大值,这里取大于０, 作激活函数
h_relu = np.maximum(h, 0)     
##print( h_relu  )
y_pred = h_relu.dot(w2)
print( y_pred  ) 

