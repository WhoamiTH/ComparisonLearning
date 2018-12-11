import numpy as np
import random
import handle_data
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