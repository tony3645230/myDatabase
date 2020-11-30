import tensorflow as tf
from ckiptagger import WS
# import numpy as np
import re
import pandas as pd
# from tensorflow.keras import backend as K


def CommentCheck(commdata):
    data = commdata
    # data = [['明星皮膚科診所', 
    #         "醫生還蠻厲害的，朋友的過敏有改善",
    #         "這間醫院醫生好厲害", "這間護士很狠", "醫師看病有耐心，對待患者很親切且熱絡。"],
    #         ['哈哈皮膚科診所', 
    #         "醫生還蠻厲害的，朋友的過敏有改善",
    #         "這間醫院醫生好厲害", "這間護士很狠", "醫師看病有耐心，對待患者很親切且熱絡。"]]

    # 獲取分析來源
    ws = WS("./data")
    print('[字元分解開始]')
    fTotal = []
    for TotalComm in data:
        TotalComm=TotalComm[1:]
        for i in TotalComm:
            fKey = re.sub(
                "[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）～]+", "", i)
            fKey = ws([fKey])
            fTotal.append(fKey[0])
            xdata = i[0:9]
            xdata = xdata.ljust(10, "．")
            xtemp = str(fTotal.index((fKey[0]))+1)
            xtotal = str(len(data)+1)
            print(f'[{xdata}] 分解完畢...... [ {xtemp.zfill(3)} / {xtotal.zfill(3)}]')
            print('[字元分解結束]')

        # 獲取分析來源
        dictionary = {}
        df = pd.read_csv("encoding.csv", encoding="utf-8")
        df = df.iloc[:]  # 選取欲讀取資料筆數
        dictionary = df.set_index('commit').T.to_dict('list')

        # 重新編譯分割後字串
        print('[字元編碼開始]')
        fileTrainSeg = []  # 編譯後資料及其原始評分
        # fileTrainSeg = np.empty(shape=(0,100))
        for text in fTotal:
            fKey = []
            for x in text:
                try:
                    fKey.append(dictionary[x][0])
                except KeyError:
                    print(x)
                    pass
            fileTrainSeg.append(fKey)
        print('[字元編碼結束]')

        data = tf.keras.preprocessing.sequence.pad_sequences(fileTrainSeg, maxlen=100)

        # 載入訓練好模型
        reload_model = tf.keras.models.load_model('saved_model/my_model')
        reload_model.summary()
        model = tf.keras.Model
        fLevel = reload_model.predict(data)
        myCore=0
        for i in fLevel:
            myCore += i.argmax() 
        myCore = myCore/len(fLevel)
        print(myCore)

# CommentCheck(123)