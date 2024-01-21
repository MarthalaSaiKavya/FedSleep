import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import datetime
import warnings
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Specify the path to your local dataset directory
local_dataset_path = r'C:\Users\smarthala\Desktop\sleep\dataset'

# Read the CSV file into a Pandas DataFrame
df_client5=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\Sleep_final\\dataset\\sleep_dataset_2001_to_2023.csv')
df_client4=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\Sleep_final\\dataset\\client1_dataset.csv')
df_client3=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\Sleep_final\\dataset\\client3_dataset.csv')
df_client2=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\Sleep_final\\dataset\\client2_dataset.csv')
df_client1=pd.read_csv('C:\\Users\\smarthala\\OneDrive - Microsoft\\Desktop\\Sleep_final\\dataset\\client1_dataset.csv')

# Feature Engineering

# I merged four files into one
df=pd.concat([df_client1, df_client2,df_client3,df_client4,df_client5])

# There are some NaNs in data. So I droped NaN.
df=df.dropna()

# I found some different type of data in some coulumns, so I fixed them.
df['HOURS OF SLEEP'] = df['HOURS OF SLEEP'].replace('6:46', '6:46:00')
df['HOURS OF SLEEP'] = df['HOURS OF SLEEP'].replace('6:06', '6:06:00')

# I changed 'DATE' to Datetime format and,
# Changed 'HOURS OF SLEEP' from timedelta64 to int, 'second'.
# I also made 'Month', 'Week' and 'Day' columns.

df['DATE']=pd.to_datetime(df['DATE'],format='%m/%d/%Y')

baseline=pd.to_datetime('00:00:00',format='%H:%M:%S')
df['HOURS OF SLEEP']=pd.to_datetime(df['HOURS OF SLEEP'],format='%H:%M:%S')-baseline
df['SECONDS OF SLEEP'] = df['HOURS OF SLEEP'].astype('int64') // 1000000000
                                    
df['Week']=df['DATE'].dt.weekday
df['Month']=df['DATE'].dt.month
df['Day']=df['DATE'].dt.day

# I changed 'REM SLEEP', 'DEEP SLEEP' and 'HEART RATE BELOW RESTING' to float.
df['REM SLEEP']=df['REM SLEEP'].str[:-1]
df['DEEP SLEEP']=df['DEEP SLEEP'].str[:-1]
df['HEART RATE BELOW RESTING']=df['HEART RATE BELOW RESTING'].str[:-1]

df['REM SLEEP']=df['REM SLEEP'].astype(float)/100
df['DEEP SLEEP']=df['DEEP SLEEP'].astype(float)/100
df['HEART RATE BELOW RESTING']=df['HEART RATE BELOW RESTING'].astype(float)/100

# I split 'SLEEP TIME' to 'Sleep_start' and 'Sleep_end' columns.
df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:21 - 8:45am', '11:21pm - 8:45am')
df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:40pm - 7:33', '11:40pm - 7:33am')
df['SLEEP TIME'] = df['SLEEP TIME'].replace('11:16pm - 7:02', '11:16pm - 7:02am')
df['SLEEP TIME'] = df['SLEEP TIME'].replace('11-38pm - 8:23am', '11:38pm - 8:23am')

df1=df['SLEEP TIME'].str.split('-', expand=True)
df1.columns = ['Sleep_start', 'Sleep_end']

df1['Sleep_start']=df1['Sleep_start'].str[:-3]
df1['Sleep_end']=df1['Sleep_end'].str[:-2]
df1['Sleep_end']=df1['Sleep_end'].str[0:]

df1['Sleep_end'] = df1['Sleep_end'].str.replace(' ', '')

df1['Sleep_start']=pd.to_datetime(df1['Sleep_start'],format='%H:%M')
df1['Sleep_end']=pd.to_datetime(df1['Sleep_end'],format='%H:%M')

df=pd.concat([df, df1],axis=1)

df['Sleep_start']=df['Sleep_start'].dt.time
df['Sleep_end']=df['Sleep_end'].dt.time

df=df.drop(['SLEEP TIME','HOURS OF SLEEP'],axis=1)

df = df.reset_index()
df=df.drop('index',axis=1)

df1=df.drop(['Day','Week','Month'],axis=1)

# I split data to 'over 80 SLEEP SCORE' =1 and 'below 80 SLEEP SCORE'=0 to classify data good or bad.
def score_judge(ex):
    if ex >= 80:
        return 1
    else:
        return 0

if 'SLEEP SCORE' in df1.columns:
    df1.loc[:, 'Evaluation'] = df1.loc[:, 'SLEEP SCORE'].apply(score_judge)
else:
    print("Column 'SLEEP SCORE' not found in DataFrame.")


# I can find the difference between evaluation 0 and 1 in 'REM SLEEP' ,'DEEP SLEEP','SECONDS OF SLEEP' and 'HEART RATE BELOW RESTING'

df2=df1.drop(['DATE', 'DAY','Sleep_start','Sleep_end','SLEEP SCORE'],axis=1)


X=df2.drop('Evaluation',axis=1).values
y=df2['Evaluation']
X_norm = (X - np.min(X)) / (np.max(X))

X_train, X_test, y_train, y_test = train_test_split(X_norm,y,test_size=0.3,random_state=42)
method_names = []
method_scores = []

# Print the number of rows in each split
print("Number of rows in X_train:", X_train.shape[0])
print("Number of rows in X_test:", X_test.shape[0])
print("Number of rows in y_train:", len(y_train))  # Assuming y is a Python list or NumPy array
print("Number of rows in y_test:", len(y_test))    # Assuming y is a Python list or NumPy array

# LOGISTIC REGRESSION

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(X_test, y_test)))
method_names.append("Logistic Reg.")
method_scores.append(log_reg.score(X_test, y_test))

# Log Loss
y_pred_proba = log_reg.predict_proba(X_test)
loss = log_loss(y_test, y_pred_proba)
print("Log Loss: {}".format(loss))

y_pred = log_reg.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.savefig('images/results.png')
plt.show()
