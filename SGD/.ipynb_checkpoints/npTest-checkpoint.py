import numpy as np
import tensorflow as tf
import torch
flag = torch.cuda.is_available()
print(flag)

def gen_line_data(sample_num=100):
    x1 = np.linspace(0, 9, sample_num)  # 生成等差数列，在0-9之间生成sample_num个数
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T  # 拼接数组
    y = np.dot(x, np.array([3, 4]).T)  # y列向量  dot点积
    return x, y


# print(tf.test.is_built_with_cuda())
