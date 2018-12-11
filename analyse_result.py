import handle_data
import predict_test
import sys
import numpy as np
import sklearn.metrics as skmet




def set_para():
    global result_number
    global winner_number
    global true_result_name
    global predict_result_name
    global record_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'result_number':
            result_number = para[1]
        if para[0] == 'winner_number':
            winner_number = para[1]
        if para[0] == 'true_result_name':
            true_result_name = para[1]
        if para[0] == 'predict_result_name':
            predict_result_name = para[1]
        if para[0] == 'record_name':
            record_name = para[1]

# -------------------------------------global parameters---------------------------------
result_number = 8
winner_number = 3
true_result_name = 'GData_test_origin.csv'
predict_result_name = 'result.csv'
record_name = 'record.txt'


# ----------------------------------set parameters---------------------------------------
set_para()



# ----------------------------------start processing-------------------------------------
true_data, true_label = handle_data.loadTrainData(true_result_name)
predict_data, predict_label = handle_data.loadTrainData(predict_result_name)
dicstart, diclength = handle_data.group(true_data)

all_group_top_precision = []
all_group_recall = []
all_group_top_exact_accuracy = []
all_group_exact_accuracy = []
all_group_auc = []

for group_index in range(len(dicstart)):
    group_start = dicstart[group_index]
    length = diclength[group_index]
    current_true_label = true_label[group_start:group_start + length]
    current_predict_label = predict_label[group_start:group_start + length]

    all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc = predict_test.analyse_group_result(current_true_label,current_predict_label,result_number,winner_number,  all_group_top_precision, all_group_recall, all_group_top_exact_accuracy,        all_group_exact_accuracy, all_group_auc)

predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, all_group_auc, record_name)