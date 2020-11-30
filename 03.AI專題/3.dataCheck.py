from re import L
from pandas.core.algorithms import mode
import tensorflow as tf
import numpy as np
from random import shuffle
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.gen_batch_ops import batch
# import re 
# import gc
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TimeDistributed


#獲取分析來源
dictionary={}
df = pd.read_csv("encoding.csv",encoding="utf-8")      
df = df.iloc[:]                                         #選取欲讀取資料筆數                                      
dictionary = df.set_index('commit').T.to_dict('list')
# print(dictionary)
#轉存編譯後數據資料及評分
df = pd.read_csv("split.csv",encoding="utf-8")
 
fileTrainSeg=[] #編譯後資料及其原始評分
i=0
for text in df['word']:
    text=text[1:-1].replace("'","")
    text=text.replace(" ","")
    text=text.split(',')
    fKey=[]
    for x in text:
        # print(dictionary[x])
        try:
            fKey.append(dictionary[x][0])
        except KeyError:
            print(x)
            pass
    fKey.append(df['label'][i])
    # print(df['label'][i])
    fileTrainSeg.append(fKey)
    i+=1
# print(fileTrainSeg)
print("編譯完成")

# Split train/test set
fileTrainSeg=np.array(fileTrainSeg)
fileTrainSeg = np.random.permutation(fileTrainSeg)
print(fileTrainSeg)
BUFFER_SIZE = int(len(fileTrainSeg)*0.8)
t_train = fileTrainSeg[:BUFFER_SIZE]
t_test = fileTrainSeg[BUFFER_SIZE:]

#資料補足長度
train_dataset = tf.keras.preprocessing.sequence.pad_sequences(t_train,maxlen=30)
test_dataset = tf.keras.preprocessing.sequence.pad_sequences(t_test,maxlen=30)


#建立訓練測試資料陣列
train_data=[]
train_label=[]
test_data=[]
test_label=[]

# train.test data/label splice 
for a in train_dataset:
    train_data.append(a[:-1])
    train_label.append(a[-1])
for a in test_dataset:
    test_data.append(a[:-1])
    test_label.append(a[-1])

# one hot label 
train_label = tf.keras.utils.to_categorical(train_label)
test_label = tf.keras.utils.to_categorical(test_label)
train_data=np.array(train_data)
test_data=np.array(test_data)

print(train_label)
print("-------------------------------")
print(train_data)


# print("-------------------------------")
# print(len(train_data[0]))
# print(len(train_label[0]))
# print(len(test_data[0]))
# print(len(test_label[0]))
# print("-------------------------------")


LenTotal = len(dictionary)+1                #字典內總資料筆數
model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=LenTotal,output_dim=64))
# model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32))
# model.add(layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(32, activation='tanh'))

# model.add(tf.keras.layers.LSTM(128,return_sequences=True))
# model.add(TimeDistributed(layers.Dense(1)))
# model.add(Dropout(0.5))
# model.add(tf.keras.layers.LSTM(128))

# model.add(layers.BatchNormalization())
# model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(Dropout(0.5))
model.add(layers.Dense(6, activation='softmax'))
model.add(layers.Dense(6, activation='sigmoid'))
# model.add(Dropout(0.2))

model.summary()

# adam = tf.keras.optimizers.Nadam(lr=0.1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(train_data,train_label,epochs=50,batch_size=32,
            validation_data = (test_data, test_label))

acc = pd.DataFrame(history.history)
acc.to_csv('saved_model/my_model2/my_model.csv',index=False)
# model.save('saved_model/my_model2')

def plot_learn_curve(history,epoch):
    epoch_range=range(1,epoch+1)
    plt.plot(epoch_range,history.history['accuracy'])
    plt.plot(epoch_range,history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(0.1)
    plt.legend(["Train","val"],loc="best")
    plt.show()

    plt.plot(epoch_range,history.history['loss'])
    plt.plot(epoch_range,history.history['val_loss'])
    plt.title("Model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Train","val"],loc="best")
    plt.show() 

plot_learn_curve(history,50)