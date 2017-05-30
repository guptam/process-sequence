import os
import argparse
import pandas as pd
import numpy as np

from dateutil.parser import parse
from datetime import datetime
import time

pd.options.mode.chained_assignment = None #to run loop quicker without warnings

name = 'helpdesk'
#name = 'bpi_12_w'

args = {
    'datadir': '../data/',
    'datafile': name + '.csv',
    'inputdir': '../input/{}/'.format(name),   
}

args = argparse.Namespace(**args)


if not os.path.isdir('../input/'):
    os.makedirs('../input/')
    
if not os.path.isdir(args.inputdir):
    os.makedirs(args.inputdir)


#----------------Define functions for features extraction---------------------
print('Define functions...')
data = pd.read_csv(args.datadir + args.datafile)
data['CompleteTimestamp'] = pd.to_datetime(data['CompleteTimestamp'], errors='coerce')

def calculateDuration(df):
    df['Duration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].shift(1)).fillna(0)
    return df

def calculateCumDuration(df):
    df['CumDuration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].iloc[0]).fillna(0) 
    #change df['CompleteTimestamp'][0] --> df['CompleteTimestamp'].iloc[0]
    return df

def calculateTimeSinceMidNight(x):
    x = str(x)
    x = time.strptime(x, "%Y-%m-%d %H:%M:%S")
    midnight = datetime.fromtimestamp(time.mktime(x)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(x))-midnight
    return timesincemidnight.seconds

def convert2seconds(x):
    x = int(x.total_seconds())
    return x

#----------------Extract features---------------------
print('\nExtract features...')
groupByCase = data.groupby(['CaseID'])

df = pd.DataFrame(columns=['CaseID', 'ActivityID', 'CompleteTimestamp', 'Duration', 'CumDuration', 'TimeSinceMidnight', 'WeekDay'])
#Loop all group and apply above functions
for case, group in groupByCase:
    group = calculateDuration(group)
    group = calculateCumDuration(group)
    group['Duration'] = group['Duration'].apply(convert2seconds)
    group['CumDuration'] = group['CumDuration'].apply(convert2seconds)
    group['TimeSinceMidnight'] = group['CompleteTimestamp'].apply(calculateTimeSinceMidNight)
    group['WeekDay'] = group['CompleteTimestamp'].dt.dayofweek
    
    df = df.append(group)

#convert to interger
for i in list(df):
    if i != 'CompleteTimestamp':
        df[i] = df[i].apply(int)

#save full data
df.to_csv(args.inputdir+'full_data.csv', index=False)


#----------------Split data---------------------
print('\nSplit data into train and test...')
groupByCase = df.groupby(['CaseID'])

num_cases = len(groupByCase)
train_size = int(num_cases*2/3)
test_size = num_cases - train_size

df_train = pd.DataFrame(columns=['CaseID', 'ActivityID', 'CompleteTimestamp', 'Duration', 'CumDuration', 'TimeSinceMidnight', 'WeekDay'])
df_test = pd.DataFrame(columns=['CaseID', 'ActivityID', 'CompleteTimestamp', 'Duration', 'CumDuration', 'TimeSinceMidnight', 'WeekDay'])

for i, (case, group) in enumerate(groupByCase):
    if i < train_size:     
        df_train = df_train.append(group)
    else:
        df_test = df_test.append(group)

df_train.to_csv(args.inputdir+'train.csv', index=False)
df_test.to_csv(args.inputdir+'test.csv', index=False)
print('\nDone!!!')