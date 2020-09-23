#-*-coding: utf-8-*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import  torch.nn.functional as f


if __name__ == "__main__":
    # time.sleep( 30 )
    print("loss function test.")

    m = nn.Softmax( )
    x = torch.randn(2, 3)
    X1 = torch.tensor([ -1556.7052,   6818.0483,   6381.7075,  -2911.1670,  -7387.9019, 6896.7964,  -8440.1230,  18116.3691,  -1918.4823, -16014.9609] )
    X1 = X1 / 10000.0
    print( X1 )
    output = m( X1 )
    print( "output: {}".format( output )  )
    input(">>>")


