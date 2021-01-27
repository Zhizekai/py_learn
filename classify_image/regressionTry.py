import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow import keras

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?",
                          comment='\t',
                          sep=" ",
                          skipinitialspace=True)

dataset = raw_dataset.copy()  # 浅拷贝
# dataset.tail()

dataset.isna().sum()  # 缺失值加和
dataset = dataset.dropna()  # 删除含有空值的行
dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')  # 添加虚拟变量

# print(dataset.tail())
###################################
# 分成测试集和训练集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# print(test_dataset[["MPG", "Cylinders", "Displacement", "Weight"]])
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# 统计
train_stats = train_dataset.describe()  # 自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）
train_stats.pop("MPG")  # 删除MPG的key值对应的value
train_stats = train_stats.transpose()

# 标签
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64,activation = 'relu',input_shape = [len( train_dataset.key())]),
        keras.layers.Dense(64,activation = 'relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)