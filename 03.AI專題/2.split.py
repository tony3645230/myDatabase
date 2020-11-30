import re 
import pandas as pd
from ckiptagger import WS
#-----------------------------------------
# 下載ckip資料集(第一次使用)
# from ckiptagger import data_utils
# data_utils.download_data_gdown("./")
#-----------------------------------------

#獲取分析來源
df = pd.read_csv("data.csv",encoding="utf-8")      
df = df.dropna(how='any')                               #丟棄評論空值
df = df.iloc[:]  
ws = WS("./data")
vocabulary_size = 50000 #資料庫最大資料量

 # 建立分詞後字典(去掉標點符號)
print('[去除符號開始...]')
fTotal=[]
for i in df['commit']:
    fKey = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）～]+", "", i)
    fKey=ws([fKey])
    fTotal.append(fKey[0])
dataset = list(zip(fTotal,df['label']))
df = pd.DataFrame(data =dataset,columns=["word","label"])
df.to_csv("split.csv",encoding="utf-8",index=False)
print('[去除符號結束]')

#建立字元對照字典
#計算文字出現頻率
print('[建立字典開始...]')
dictionary = {}
for text in fTotal:
    # print(text)
    for x in text:
        if x not in dictionary.keys():
            dictionary[x] = 1
        else:
            dictionary[x] += 1
sorted_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[0], reverse=True)}
print('[建立字典結束]')

df = pd.DataFrame(data = sorted_dict.items(),columns=["commit","label"])
fileTrainSeg=[] #編譯後資料及其原始評分

def embeded_word(df):
    a = {}
    for word in df['commit']:
        a[word] = len(a) + 1
    return a
dictionary=(embeded_word(df))

df = pd.DataFrame(data = dictionary.items(),columns=["commit","label"])
df.to_csv("encoding.csv",encoding="utf-8",index=False)

# df = [('皮膚癢來看病，吃藥兩天就好了！', '5'), 
# ('掛號時有說明要點痣，但護士並沒有多說什麼直到進診間時醫師才告知點痣需自費', '1')]
