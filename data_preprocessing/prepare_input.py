import os
import argparse
import pandas as pd
import numpy as np
import pickle
import sys

sys.path.insert(0, './../utils/')

from utils_2 import * #remember to change this one!!!!!

#----------------Load data---------------------
print('Loading data...')
name = 'helpdesk'
#name = 'bpi_12_w'
parser = {
    'datafile': name + '.csv',
    'inputdir': './../input/{}/'.format(name),   
    'utils': 'utils_2'
}

dirs = argparse.Namespace(**parser)

data = pd.read_csv(dirs.inputdir+'full_data.csv')
train = pd.read_csv(dirs.inputdir+'train.csv')
test = pd.read_csv(dirs.inputdir+'test.csv')

data = transformDf(data)
train = transformDf(train)
test = transformDf(test)

print('Data: {0}|Train: {1}|Test:{2}'.format(data.shape[0], train.shape[0], test.shape[0]))

#----------------Parameters---------------------
print('\nCompute parameters...')
groupByCase = data.groupby(['CaseID'])

# define the denominator for normalization
divisor = data['Duration'].mean()
divisor2 = data['CumDuration'].mean()

# find len of longest case
maxlen = findLongestLength(groupByCase)
print('Length of longest case: {}'.format(maxlen))

# define number of features
if dirs.utils == 'utils':
    features = ['number_of_past_activitiy', 'duration', 'cumduration', 'time_from_midnight', 'day_of_week']
elif dirs.utils == 'utils_1':
    features = ['number_of_past_activitiy', 'duration', 'cumduration', 'time_from_midnight', 
              'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
else:
    features=['number_of_past_activitiy', 'duration', 'time_from_midnight', 
              'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
num_features = len(data['ActivityID'].unique()) + len(features)
print('Use utility function: {}'.format(dirs.utils))
print('Features: {}'.format(features))
print('Number of features: {}'.format(num_features))

# mapping dict
unique_chars = data['ActivityID'].unique().tolist()
target_chars = unique_chars + ['EOS']
chartoindice = char_indice_dict(unique_chars)
targetchartoindice = char_indice_dict(target_chars)

#----------------Data preprocessing---------------------
print('\nPrepare data for training...')
# Train
train_groupByCase = train.groupby(['CaseID'])
# Input
sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4 = getFeature(train_groupByCase)
X = vectorizeInput(sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4, maxlen, num_features, chartoindice, divisor, divisor2, divisor3=86400, divisor4=7)
# Output
next_chars, next_chars_t, next_chars_t2, next_chars_t3, next_chars_t4 = getOutput(train_groupByCase)
y_a = one_hot_encode(next_chars, targetchartoindice)
y_t = normalize(next_chars_t, divisor)

print('Prepare data for testing...')
# Test
test_groupByCase = test.groupby(['CaseID'])
df_test = pd.DataFrame(columns=['CaseID', 'ActivityID', 'CompleteTimestamp', 'Duration', 'CumDuration', 'TimeSinceMidnight', 'WeekDay'])
# only consider the case has at least 2 activities
for case, group in test_groupByCase:
    if group.shape[0] > 1:     
        df_test = df_test.append(group)
print('Number of cases with less than 2 activity in test set: {}'.format(test.shape[0]-df_test.shape[0]))
# Input
sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4 = getFeature(test_groupByCase)
X_test = vectorizeInput(sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4, maxlen, num_features, chartoindice, divisor, divisor2)
# Output
next_chars, next_chars_t, next_chars_t2, next_chars_t3, next_chars_t4 = getOutput(test_groupByCase)
y_a_test = one_hot_encode(next_chars, targetchartoindice)
y_t_test = normalize(next_chars_t, divisor)

#----------------Save data---------------------
print('\nSave data...')
with open(dirs.inputdir + 'parameters.pkl', 'wb') as f:
    pickle.dump(maxlen, f, protocol=2)
    pickle.dump(num_features, f, protocol=2)
    pickle.dump(chartoindice, f, protocol=2)
    pickle.dump(targetchartoindice, f, protocol=2)
    pickle.dump(divisor, f, protocol=2)
    pickle.dump(divisor2, f, protocol=2)

with open(dirs.inputdir + 'preprocessed_data.pkl', 'wb') as f:
    pickle.dump(X, f, protocol=2)
    pickle.dump(y_a, f, protocol=2)
    pickle.dump(y_t, f, protocol=2)
    pickle.dump(X_test, f, protocol=2)
    pickle.dump(y_a_test, f, protocol=2)
    pickle.dump(y_t_test, f, protocol=2)

print('\nDone!!!')