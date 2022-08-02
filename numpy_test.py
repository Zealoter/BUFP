"""
# @Author: JuQi
# @Time  : 2022/7/27 16:11
# @E-mail: 18672750887@163.com
"""
import numpy as np

a = np.array([[0, 1], [2, 2], [4, 3]])

b= (a == np.max(a,axis=1)[:,None]).astype(int)
# b=(a == a.max(axis=1)[:,None]).astype(int)
print(b)
