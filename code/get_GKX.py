import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import svd

os.chdir("/home/sn576/project/EAP_GKX_Replication/data")

def fill_na(data_ch, characteristics):
    for ch in characteristics:
         data_ch[ch] = data_ch.groupby('DATE')[ch].transform(lambda x: x.fillna(x.median()))
    for ch in characteristics:
         data_ch[ch] = data_ch[ch].fillna(0)
    return data_ch

# get dummies for SIC code
def get_sic_dummies(data_ch):
    sic_dummies = pd.get_dummies(data_ch['sic2'].fillna(999).astype(int),prefix='sic')
    if 'sic_999' in sic_dummies:
      sic_dummies.drop('sic_999',axis=1)
    data_ch_d = pd.concat([data_ch,sic_dummies],axis=1)
    data_ch_d.drop(['prc','SHROUT','mve0','sic2'],inplace=True,axis=1)
    return data_ch_d

stdt, nddt = "1957-01-01", "2016-12-31"

# load firm characteristics data
sset = "all"
data_ch = pd.read_csv('GKX_20201231_' + sset + '.csv')
if 'Unnamed: 0' in data_ch.columns:
    # Remove the unnamed column
    data_ch = data_ch.drop('Unnamed: 0', axis=1)
# For now do with 100
# data_ch = data_ch.sort_values('mvel1',ascending=False).groupby('DATE').head(100).reset_index(drop=True)
data_ch['DATE'] = pd.to_datetime(data_ch['DATE'],format='%Y%m%d')+pd.offsets.MonthEnd(0)
data_ch = data_ch[(data_ch['DATE']>=stdt)&(data_ch['DATE']<=nddt)].reset_index(drop=True)
characteristics = list(set(data_ch.columns).difference({'permno','DATE','SHROUT','mve0','sic2','RET','prc'}))

data_ch.head()

data_ch = fill_na(data_ch, characteristics)
data_ch.shape

data_ch = get_sic_dummies(data_ch)
data_ch.shape
print(data_ch.columns)

# load macroeconomic predictors data
data_ma = pd.read_csv('PredictorData2021.csv')
data_ma['yyyymm'] = pd.to_datetime(data_ma['yyyymm'],format='%Y%m')+pd.offsets.MonthEnd(0)
data_ma = data_ma[(data_ma['yyyymm']>= stdt) & 
                  (data_ma['yyyymm']<= nddt)].reset_index(drop=True)

# construct predictor
ma_predictors = ['dp_sp','ep_sp','bm_sp','ntis','tbl','tms','dfy','svar']
data_ma['dp_sp'] = data_ma['D12']/data_ma['Index']
data_ma['ep_sp'] = data_ma['E12']/data_ma['Index']
data_ma.rename({'b/m':'bm_sp'},axis=1,inplace=True)
data_ma['tms'] = data_ma['lty']-data_ma['tbl']
data_ma['dfy'] = data_ma['BAA']-data_ma['AAA']
data_ma = data_ma[['yyyymm']+ma_predictors]

data_ma.head()
features = list(set(data_ch.columns).difference({'permno','DATE','RET'}))
df = pd.merge(data_ch,data_ma,left_on='DATE',right_on='yyyymm',how='left')
df = df.drop(columns=['yyyymm'])
# stdt_vld = np.datetime64('1975-01-31')
# stdt_tst = np.datetime64('1987-01-31')
stdt_vld = np.datetime64('2010-01-31')
stdt_tst = np.datetime64('2012-01-31')

def trn_vld_tst(data, stdt_vld, stdt_tst, features):
    nddt_tst = stdt_tst.astype('datetime64[Y]') + np.timedelta64(1, 'Y')
    # training setstdt_vld = np.datetime64('2001-01-31')
    X_trn = data[data['DATE']<stdt_vld][features]
    y_trn = data[data['DATE']<stdt_vld]["RET"]
    # validation set
    X_vld = data[(data['DATE']<stdt_tst)&(data['DATE']>=stdt_vld)][features]
    y_vld = data[(data['DATE']<stdt_tst)&(data['DATE']>=stdt_vld)]["RET"]
    # testing set
    X_tst = data[(data['DATE'] < nddt_tst) & (data['DATE'] >= stdt_tst)][features]
    y_tst = data[(data['DATE'] < nddt_tst) &  (data['DATE'] >= stdt_tst)]["RET"]
    print(f"The shape of the training set is: {X_trn.shape}")
    print(f"The shape of the validation set is: {X_vld.shape}")
    print(f"The shape of the test set is: {X_tst.shape}")
    return X_trn, X_vld, X_tst, y_trn, y_vld, y_tst

X_trn, X_vld, X_tst, y_trn, y_vld, y_tst = trn_vld_tst(df, stdt_vld, stdt_tst, features)

