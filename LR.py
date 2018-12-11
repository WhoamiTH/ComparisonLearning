# handle data head

import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib

# handle data function begining

def loadTrainData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    # data = tem_data.astype(np.float).astype(np.int)
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label

def loadTestData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',')
    column_name = tem[0,0:25]
    first_column = tem[1:,1]
    tem_data = tem[1:, 1:25]
    data = tem_data.astype(np.float)
    return data, column_name, first_column


def group(Data):
    i = 0
    j = 1
    l = 1
    t = Data[0][2]
    Ds = {}
    Dl = {}
    Ds[0] = 0
    while j < len(Data):
        if (t != Data[j][2]):
            Dl[i] = l
            Ds[i + 1] = j
            i += 1
            l = 1
            t = Data[j][2]
            j += 1
        elif (j == len(Data) - 1):
            Dl[i] = l + 1
            j += 1
        else:
            l += 1
            j += 1
    return Ds, Dl

def data_extend(Data_1, Data_2):
    m = list(Data_1)
    n = list(Data_2)
    return m + n

def condense_data_pca(Data, num_of_components):
    pca = PCA(n_components=num_of_components)
    pca.fit(Data)
    return pca


def condense_data_kernel_pca(Data, num_of_components):
    kernelpca = KernelPCA(n_components=num_of_components)
    kernelpca.fit(Data)
    return kernelpca


def standardize_data(Data):
    scaler = skpre.StandardScaler()
    scaler.fit(Data)
    return scaler


def standarize_PCA_data(Data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name):
    scaler = standardize_data(Data)
    new_data = scaler.transform(Data)
    if pca_or_not :
        pca = condense_data_pca(new_data, num_of_components)
        new_data = pca.transform(new_data)
        joblib.dump(pca, pca_name)
    if kernelpca_or_not :
        kernelpca = condense_data_kernel_pca(new_data, num_of_components)
        new_data = kernelpca.transform(new_data)
        joblib.dump(kernelpca, kernelpca_name)
    joblib.dump(scaler, scaler_name)
    return new_data

def transform_data_by_standarize_pca(Data, scaler_name, pca_name, kernelpca_name):
    scaler = joblib.load(scaler_name)
    new_data = scaler.transform(Data)
    # copy
    if pca_name:
        pca = joblib.load(pca_name)
        new_data = pca.transform(new_data)
    if kernelpca_name:
        kernelpca = joblib.load(kernelpca_name)
        new_data = kernelpca.transform(new_data)
    return new_data



def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def generate_primal_train_data(Data,Label,Ds,Dl,num_of_train):
    # train_index_start = random.randint(0,len(Ds)-num_of_train)
    train_index_start = 0
    front = Ds[train_index_start]
    end = Ds[train_index_start+num_of_train-1]+Dl[train_index_start+num_of_train-1]
    train_x = Data[front:end,:]
    train_y = Label[front:end]
    return train_index_start,train_x,train_y


def handleData_extend_mirror(Data, Label, positive_value, negative_value):
    temd = []
    teml = []
    length = len(Data)
    for j in range(length):
        for t in range(length):
            if j != t:
                temd.append(data_extend(Data[j], Data[t]))
                if Label[j] > Label[t]:
                    # teml.append([-1])
                    teml.append([negative_value])
                else:
                    teml.append([positive_value])
    return temd, teml


def handleData_extend_not_mirror(Data, Label, positive_value, negative_value):
    temd = []
    teml = []
    length = len(Data)
    for j in range(length):
        for t in range(j+1,length):
            temd.append(data_extend(Data[j], Data[t]))
            if Label[j] > Label[t]:
                teml.append([negative_value])
            else:
                teml.append([positive_value])
    return temd, teml


def transform_data_to_compare_data(Data, Label, Ds, Dl, mirror_type, positive_value, negative_value):
    tem_data = []
    tem_label = []
    for group_index in range(len(Ds)):
        group_start = Ds[group_index]
        length = Dl[group_index]
        current_group_data = Data[group_start:group_start+length,:]
        current_group_label = Label[group_start:group_start+length]
        if mirror_type == 'mirror':
            temd, teml = handleData_extend_mirror(current_group_data, current_group_label, positive_value, negative_value)
        else:
            temd, teml = handleData_extend_not_mirror(current_group_data, current_group_label, positive_value, negative_value)
        tem_data = tem_data + temd
        tem_label = tem_label + teml

    data = np.array(tem_data)
    label = np.array(tem_label)

    return data, label


def digit(x):
    if str.isdigit(x) or x == '.':
        return True
    else:
        return False

def alpha(x):
    if str.isalpha(x) or x == ' ':
        return True
    else:
        return False

def point(x):
    return x == '.'

def divide_digit(x):
    d = filter(digit, x)
    item = ''
    for i in d:
        item += i
    if len(item) == 0:
        return 0.0
    else:
        p = filter(point, item)
        itemp = ''
        for i in p:
            itemp += i
        # print(itemp)
        if len(itemp) > 1:
            return 0.0
        else:
            return float(item)

def divide_alpha(x):
    a = filter(alpha, x)
    item = ''
    for i in a:
        item += i
    return item

def divide_alpha_digit(x):
    num = divide_digit(x)
    word = divide_alpha(x)
    return word,num

def initlist():
    gp = []
    gr = []
    ga = []
    agtp = []
    agr = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp,gr,ga,agtp,agr,agtea,aga,tt,rt

def aver(l):
    return sum(l)/len(l)

def scan_file(file_name):
    f = open(file_name,'r')
    gp,gr,ga,agtp,agr,agtea,aga,tt,rt = initlist()
    for i in f:
        word,num = divide_alpha_digit(i)
        if word == 'the average group top precision is ':
            agtp.append(num)
        if word == 'the average group recall is ':
            agr.append(num)
        if word == 'the average group top exact accuracy is ':
            agtea.append(num)
        if word == 'the average group accuracy is ':
            aga.append(num)
        if word == 'the  time training time is ':
            tt.append(float(str(num)[1:-1]))
        if word == 'the  time running time is ':
            rt.append(float(str(num)[1:-1]))
    av_aptp = aver(agtp)
    av_agr = aver(agr)
    av_agtea = aver(agtea)
    av_aga = aver(aga)
    av_tt = aver(tt)
    av_rt = aver(rt)
    return av_aptp,av_agr,av_agtea,av_aga,av_tt,av_rt

def append_file(file_name):
    av_agtp, av_agr, av_agtea, av_aga, av_tt, av_rt = scan_file(file_name)
    fscore = (2*av_agtp*av_agr)/(av_agtp+av_agr)
    f = open(file_name,'a')
    f.write("the F-score is {0}\n".format(fscore))
    f.write("the average group top precision is {0}\n".format(av_agtp))
    f.write("the average group recall is {0}\n".format(av_agr))
    f.write("the average group top exact accuracy is {0}\n".format(av_agtea))
    f.write("the average group accuracy is {0}\n".format(av_aga))
    f.write("the 3 time training time is {0}\n".format(av_tt))
    f.write("the 3 time running time is {0}\n".format(av_rt))
    f.close()

# handle data functions ending




# --------------------------------------------------------------------------------------




# predict test head

# predict test function begining

import numpy as np
import random

import sklearn.metrics as skmet
import time


def change_to_0_1(label,num):
    length_of_label = len(label)
    if length_of_label <= num:
        tem = [1 for t in range(length_of_label)]
    else:
        tem = [0 for t in range(length_of_label)]
        for i in range(num):
            tem[label[i]-1] = 1
    return tem


def calacc(true_label, predict_label, result_number, winner_number):
    changed_rank = change_to_0_1(true_label, winner_number)
    changed_label = change_to_0_1(predict_label,result_number)

    en = 0
    if len(changed_rank) != len(changed_label):
        print(true_label)
        print(predict_label)
        print(changed_rank)
        print(changed_label)
        time.sleep(5)
    for i in range(len(changed_rank)):
        if changed_rank[i] == changed_label[i]:
            en += 1
    return en / len(true_label)

def count_top(y_true, y_pred, result_number, winner_number):
    tp = 0
    exact = 0
    if result_number <= len(y_true):
        top_true = y_true[:winner_number]
        top_pred = y_pred[:result_number]
    elif len(y_true) >= winner_number:
        top_true = y_true[:winner_number]
        top_pred = y_pred
    else:
        top_true = y_true
        top_pred = y_pred
    len_top = len(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if i < winner_number:
                if top_pred[i] == top_true[i]:
                    exact += 1
    if result_number == len_top:
        group_pre = tp/result_number
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    elif len_top >= winner_number:
        group_pre = tp/len_top
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    else:
        group_pre = tp/len_top
        group_recall = tp/len_top
        group_top_exact_accuracy = exact/len_top
    return group_pre, group_recall, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

def rank_the_group(group_data, reference, model, threshold):
    tem = [reference.pop()]
    for each in reference:
        for item in range(len(tem)):
            t = handle_data.data_extend(group_data[each-1], group_data[tem[item]-1])
            t = np.array(t).reshape((1,-1))
            if model.predict(t) > threshold:
                tem.insert(item, each)
                break
            else:
                if item == len(tem)-1:
                    tem.append(each)
                    break
    return tem

def group_test(Data, model, threshold_value):
    length = len(Data)
    reference = [t for t in range(1, length + 1)]
    # random.shuffle(reference)
    predict_rank = rank_the_group(Data, reference, model, threshold_value)
    predict_rank = handle_data.exchange(predict_rank)
    return predict_rank


def analyse_group_result(true_label, predict_label, result_number, winner_number, all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc):
    group_true_label = handle_data.exchange(true_label)
    group_predict_label = handle_data.exchange(predict_label)
    group_top_precision, group_recall, group_top_exact_accuracy = count_top(group_true_label, group_predict_label, result_number, winner_number)

    group_exact_accuracy = calacc(group_true_label, group_predict_label, result_number, winner_number)

    change_true_label = change_to_0_1(group_true_label, winner_number)
    change_predict_label = change_to_0_1(group_predict_label, result_number)
    change_true_label = np.array(change_true_label)
    change_predict_label = np.array(change_predict_label)
    
    all_group_auc.append(skmet.roc_auc_score(change_true_label, change_predict_label))
    all_group_top_precision.append(group_top_precision)
    all_group_recall.append(group_recall)
    all_group_top_exact_accuracy.append(group_top_exact_accuracy)
    all_group_exact_accuracy.append(group_exact_accuracy)
    return all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc


def cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_accuracy, all_group_auc, record_name):
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_recall = sum(all_group_recall)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    average_group_auc = sum(all_group_auc)/totle
    fscore = (2 * average_group_top_precision * average_group_recall) / (
                average_group_top_precision + average_group_recall)

    print("the AUC is {0}\n".format(average_group_auc))
    print("the Fscore is {0}\n".format(fscore))
    print("the average group top precision is {0}\n".format(average_group_top_precision))
    print("the average group recall is {0}\n".format(average_group_recall))
    print("the average group top exact accuracy is {0}\n".format(average_group_top_exact_accuracy))
    print("the average group accuracy is {0}\n".format(average_group_accuracy))

    record = open(record_name,'w')
    record.write("the AUC is {0}\n".format(average_group_auc))
    record.write("the Fscore is {0}\n".format(fscore))
    record.write("the average group top precision is {0}\n".format(average_group_top_precision))
    record.write("the average group recall is {0}\n".format(average_group_recall))
    record.write("the average group top exact accuracy is {0}\n".format(average_group_top_exact_accuracy))
    record.write("the average group accuracy is {0}\n".format(average_group_accuracy))

# predict test functions ending 




# -------------------------------------------------------------------------------------




# train processing head

%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from sklearn.externals import joblib
from time import clock

import sys

# train processing

# -------------------------------------parameters----------------------------------------
# model_type = 'LR'
model_type = 'SVC'
# model_type = 'DT'
# mirror_type = "mirror"
mirror_type = "not_mirror"
positive_value = 1
negative_value = -1
threshold_value = 0

kernelpca_or_not = False
pca_or_not = True
num_of_components = 19

scaler_name = 'scaler.m'

pca_name = 'pca.m'
kernelpca_name = 'kernelpca.m'
model_name = 'model.m'

file_name = 'train.csv'

# data input

data, label = loadTrainData(file_name)

# divided data into different group

dicstart, diclength = group(data)

# record the start time

start = clock()

# unsing PCA to reduce the dimension

new_data = standarize_PCA_data(data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)

#transform the data to a new format

train_data,train_label= transform_data_to_compare_data(new_data, label, dicstart, diclength, mirror_type, positive_value, negative_value)

# create LogisticRegression model

single_input_size = 19

transformed_input_size = single_input_size * 2

num_class = 1

x = tf.placeholder(tf.float32, [None, transformed_input_size])

y_true = tf.placeholder(tf.int64, [None])


weights = tf.Variable(tf.zeros([transformed_input_size, num_class]))

biases = tf.Variable(tf.zeros([num_class]))


# method one

# y_pred = tf.sigmod(tf.matmul(x, weights) + biases)

# loss = -y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)


# method second
# This y_pred is just the intermediate result which is required sigmod function
y_pred = tf.matmul(x, weights) + biases

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)


cost = rf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# create session and run the model

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	feed_dict_train = {
					x  		: train_data,
					y_true 	: train_label
	}

	for i in range(100):
		cost_val, opt_obj = sess.run( [cost, optimizer], feed_dict=feed_dict_train )
		print('cost = {}'.format(cost_val))