import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

filepath1 = "Titanic_survive/train.csv"
filepath2 = "Titanic_survive/test.csv"


train_df = pd.read_csv(filepath1)  #读取csv文件

def data_preprocessing(all_df,flag):
    # 选择需要使用的数据属性
    if flag == 1:
        cols = ['Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    else:
        cols = ['Survived','Name','Pclass','Sex','Age','SibSp','Parch','Fare']
    all_df = all_df[cols]
    df = all_df.drop(['Name'], axis=1) #删掉姓名

    #查看数据缺省值情况
    #print(all_df.isnull().sum())

    #对存在缺省值的项目使用平均值代替
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)

    #将性别转换为逻辑值
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    return df


train_data = data_preprocessing(train_df,flag=0)

train_y = train_data['Survived']
train_x = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare']]

#训练数据归一化处理
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
scaled_train_x = minmax_scale.fit_transform(train_x)

print(scaled_train_x[:2])

#使用逻辑回归进行模型训练
log_reg = LogisticRegression(C=10)   # C为正则化系数
log_reg.fit(scaled_train_x, train_y)

test_df = pd.read_csv(filepath2)  #读取csv文件
id = test_df['PassengerId']
test_data = data_preprocessing(test_df,flag=1)

#测试数据归一化处理
scaled_test_x = minmax_scale.fit_transform(test_data)


y_predict = log_reg.predict(scaled_test_x)

yyy = pd.DataFrame(data=y_predict.flatten())
idd = pd.DataFrame(data=id)


pd.concat([idd,yyy],axis = 1).to_csv(path_or_buf="Titanic_survive/log_reg_submission.csv",
                                      header=['PassengerId','Survived'],index=False)
