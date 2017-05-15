import pandas as pd
import numpy as np


def transformDf(data):
    '''This function returns a dataframe with the type of activity is a string and the other are float'''
    # convert to float
    for col in list(data):
        if col != 'CompleteTimestamp':
            data[col] = data[col].apply(float)

    # convert activity from float to str
    data.ActivityID = data.ActivityID.astype(str)
    
    return data
    
def findLongestLength(groupByCase):
    '''This function returns the length of longest case'''
    #groupByCase = data.groupby(['CaseID'])
    maxlen = 1
    for case, group in groupByCase:
        temp_len = group.shape[0]
        if temp_len > maxlen:
            maxlen = temp_len
    maxlen += 1 # include EOS
    return maxlen

def char_indice_dict(char_list):
    '''This function returns a mapping dictionary'''
    chartoindice = {}
    for indice, char in enumerate(char_list):
        chartoindice[char] = indice
    return chartoindice

def getList(df):
    '''This function takes a pandas series as an input and returns a sequence of elements of that series'''
    temp = []
    lst = df.tolist()
    for i in range(1, len(lst)+1):
        sub_lst = lst[:i]
        temp.append(sub_lst)
    return temp

def getFeature(groupByCase):
    '''This function returns a sequence of all cases'''
    sentences = []      #activity
    sentences_t = []    #duration
    sentences_t2 =[]    #cum duration
    sentences_t3 = []   #time since midnight
    sentences_t4 = []   #weekday
    for case, group in groupByCase:
        case_sentences = getList(group['ActivityID'])
        sentences += case_sentences
    
        case_sentences_t = getList(group['Duration'])
        sentences_t += case_sentences_t
    
        case_sentences_t2 = getList(group['CumDuration'])
        sentences_t2 += case_sentences_t2
    
        case_sentences_t3 = getList(group['TimeSinceMidnight'])
        sentences_t3 += case_sentences_t3
    
        case_sentences_t4 = getList(group['WeekDay'])
        sentences_t4 += case_sentences_t4
    return sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4


def vectorizeInput(groupByCase, maxlen, num_features, chartoindice, divisor, divisor2, divisor3=86400, divisor4=7):
    '''This function returns a vectorized input'''
    sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4 = getFeature(groupByCase)
    X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen-len(sentence)
        unique_chars = chartoindice.keys()
        sentence_t = sentences_t[i]
        sentence_t2 = sentences_t2[i]
        sentence_t3 = sentences_t3[i]
        sentence_t4 = sentences_t4[i]
        for t, char in enumerate(sentence):
            #fill activity
            for c in unique_chars:
                if c==char:
                    X[i, t+leftpad, chartoindice[c]] = 1

            #fill time  
            X[i, t+leftpad, len(unique_chars)] = t+1
            X[i, t+leftpad, len(unique_chars)+1] = sentence_t[t]/divisor
            X[i, t+leftpad, len(unique_chars)+2] = sentence_t2[t]/divisor2
            X[i, t+leftpad, len(unique_chars)+3] = sentence_t3[t]/divisor3
            X[i, t+leftpad, len(unique_chars)+4] = sentence_t4[t]/divisor4
    return X

def getNextActivity(df):
    '''This is used to get next activity'''
    temp = []
    lst = df.tolist()
    for i in range(1, len(df)):
        ele = lst[i]
        temp.append(ele)
    temp.append('EOS') #EOS: end of sentence
    return temp

def getNextTime(df):
    '''This is used to get next time'''
    temp = []
    lst = df.tolist()
    for i in range(1, len(df)):
        ele = lst[i]
        temp.append(ele)
    temp.append(0) # beginning time of next activity
    return temp

def getOutput(groupByCase):
    next_chars = []
    next_chars_t = []
    next_chars_t2 = []
    next_chars_t3 = []
    next_chars_t4 = []

    for case, group in groupByCase:
        case_next_char = getNextActivity(group['ActivityID'])
        next_chars += case_next_char
    
        case_next_char_t = getNextTime(group['Duration'])
        next_chars_t += case_next_char_t
    
        case_next_char_t2 = getNextTime(group['CumDuration'])
        next_chars_t2 += case_next_char_t2
    
        case_next_char_t3 = getNextTime(group['TimeSinceMidnight'])
        next_chars_t3 += case_next_char_t3
    
        case_next_char_t4 = getNextTime(group['WeekDay'])
        next_chars_t4 += case_next_char_t4
    
    return next_chars, next_chars_t, next_chars_t2, next_chars_t3, next_chars_t4


def one_hot_encode(groupByCase, targetchartoindice):
    '''This function returns a one-hot-encoded y_a'''
    next_chars = getOutput(groupByCase)[0]
    target_chars = targetchartoindice.keys()

    y_a = np.zeros((len(next_chars), len(target_chars)), dtype=np.float32)
    for i in range(len(next_chars)):
        for c in target_chars:
            if c==next_chars[i]:
                y_a[i, targetchartoindice[c]] = 1
    return y_a

def nomalize(groupByCase, divisor):
    next_chars_t = getOutput(groupByCase)[1]
    y_t = np.asarray(next_chars_t)
    return y_t/divisor
'''
# new next_chars
next_chars_indice = [targetchartoindice[act] for act in next_chars]
# reshape for OHC without warning
next_chars_indice = np.asarray(next_chars_indice).reshape(-1,1)

next_chars_indice[:10]

encoder = OneHotEncoder()
data_feature_one_hot_encoded = encoder.fit_transform(next_chars_indice)

y_a = data_feature_one_hot_encoded.toarray()
y_a

#y_a.shape (13710, 9)
'''

'''
next_chars_t = next_chars_t.reshape(-1, 1)
scaler = StandardScaler().fit(next_chars_t)
y_t = scaler.transform(next_chars_t) 
y_t = y_t.reshape([13710,])
'''