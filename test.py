from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import sys
import numpy as np
import csv

def set_para():
    global model_type
    global mirror_type
    global scaler_name
    global kernelpca_name
    global pca_name
    global model_name
    global record_name
    global file_name


    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]
        if para[0] == 'record_name':
            record_name = para[1]
        if para[0] == 'file_name':
            file_name = para[1]
    if kernelpca_name and pca_name:
        kernelpca_name = ''



# -------------------------------------global parameters---------------------------------
threshold_value = 0
scaler_name = 'scaler.m'
kernelpca_name = ''
pca_name = 'pca.m'
model_name = 'model.m'
file_name = 'GData_test.csv'
record_name = 'result.csv'
# ----------------------------------set parameters---------------------------------------
set_para()

# ----------------------------------start processing-------------------------------------
print(file_name)
data, column_name, first_column = handle_data.loadTestData(file_name)
dicstart, diclength = handle_data.group(data)
model = joblib.load(model_name)
test_group_num = len(dicstart)

all_result = []
start = clock()
new_data = handle_data.transform_data_by_standarize_pca(data, scaler_name, pca_name, kernelpca_name)
for group_index in range(test_group_num):
    print('the {0} group race'.format(group_index+1))
    group_start = dicstart[group_index]
    length = diclength[group_index]
    current_group_data = new_data[group_start:group_start + length, :]
    all_result += predict_test.group_test(current_group_data, model, threshold_value)
finish = clock()

all_result = np.array(all_result)

column_name = column_name.tolist()
column_name.append('Rank')
column_name = np.array(column_name).reshape(1,-1)

result_content = first_column.reshape(-1,1)
result_content = np.hstack((result_content, data))
all_result = all_result.reshape(-1,1)
result_content = np.hstack((result_content, all_result))
result_content = result_content.astype(np.str)

result_content = np.vstack((column_name, result_content))

with open(record_name,"w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result_content)
print('Done')