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


def vectorizeInput(sentences, sentences_t, sentences_t2, sentences_t3, sentences_t4, maxlen, num_features, chartoindice, divisor, divisor2, divisor3=86400, divisor4=7):
    '''This function returns a vectorized input'''
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


def one_hot_encode(next_chars, targetchartoindice):
    '''This function returns a one-hot-encoded y_a'''
    target_chars = targetchartoindice.keys()

    y_a = np.zeros((len(next_chars), len(target_chars)), dtype=np.float32)
    for i in range(len(next_chars)):
        for c in target_chars:
            if c==next_chars[i]:
                y_a[i, targetchartoindice[c]] = 1
    return y_a

def normalize(next_chars_t, divisor):
    y_t = np.asarray(next_chars_t)
    return y_t/divisor

def getLabel(prediction, targetchartoindice):
    indicetotargetchar = dict((indice, char) for char, indice in targetchartoindice.items())
    label_list = []
    for i in range(prediction.shape[0]):
        max_value_index = np.argmax(prediction[i])
        label = indicetotargetchar[max_value_index]
        label_list.append(label)
    return label_list


def inverseTime(predictions, divisor):
    pred_t = predictions*divisor
    return np.maximum(pred_t, 0)


def get_top3_accuracy(probabilities_array, actual_labels, targetchartoindice):
    match = 0.0
    total = len(actual_labels)
    for i in range(len(probabilities_array)):
        current_probabilites = probabilities_array[i]
        top_pred_labels = get_top3_labels(current_probabilites, targetchartoindice) 
        if actual_labels[i] in top_pred_labels:
            match +=1
    return match/total

def get_top3_labels(current_probabilites, targetchartoindice):
    labels = []
    indicetotargetchar = dict((indice, char) for char, indice in targetchartoindice.items())
    top_3_index = np.argpartition(-current_probabilites, 3)[:3]
    for i in range(len(top_3_index)):
        pred_label = indicetotargetchar[top_3_index[i]]
        labels.append(pred_label)
    return labels