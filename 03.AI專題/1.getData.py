import mysql.connector
import pandas as pd

from pandas.core.indexes.base import Index 
from ckiptagger import WS

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="hos"
    )
mycursor = mydb.cursor()
sql = 'SELECT comment,rating FROM hos'

mycursor.execute(sql)
fileTrainRead = mycursor.fetchall()
commit=[]
label=[]
for i in fileTrainRead:
    commit.append(i[0])
    label.append(i[1])
dataset = list(zip(commit,label))
df = pd.DataFrame(data = dataset,columns=["commit","label"])
df.to_csv("data.csv",encoding="utf-8",index=False)

#-----------------------------------------------------------------------------

