import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

filepath1 = "titanic/train.csv"
filepath2 = "titanic/test.csv"


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

#使用keras建立模型

model = Sequential()

model.add(Dense(units=40, input_dim=6,kernel_initializer='uniform',activation='relu' ))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform',activation='sigmoid'))

#编译并训练模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history = model.fit(x=scaled_train_x, y=train_y, validation_split=0.1, epochs=30, batch_size=30, verbose= 2)

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history,'acc','val_acc')

test_df = pd.read_csv(filepath2)  #读取csv文件
id = test_df['PassengerId']
test_data = data_preprocessing(test_df,flag=1)

#测试数据归一化处理
scaled_test_x = minmax_scale.fit_transform(test_data)

y_predict = model.predict_classes(scaled_test_x)


yyy = pd.DataFrame(data=y_predict.flatten())
idd = pd.DataFrame(data=id)


pd.concat([idd,yyy],axis = 1).to_csv(path_or_buf="titanic/NN_submission3.csv",
                                      header=['PassengerId','Survived'],index=False)