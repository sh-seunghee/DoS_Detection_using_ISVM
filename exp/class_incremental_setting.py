from __future__ import division
import pandas as pd
import numpy as np
import time
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

'''
Import Data
'''

files = glob.glob("../MachineLearningCVE/*.csv")
df = pd.DataFrame()

for file in files:

    cont = pd.read_csv(file)
    df = df.append(cont)

'''

# Handling "Infinity" and "Nan";

'''

# First, delete rows that has infinity value - Get names of indexes for the columns has value infinity
idxNames = df[df["Flow Bytes/s"] == "Infinity"].index
df = df.drop(idxNames, inplace=False)

idxNames = df[df[" Flow Packets/s"] == "Infinity"].index
df = df.drop(idxNames, inplace=False)

# Second, modify Nan to zero
df = df.replace("NaN", 0)

# Drop the columns that has the same value in all samples
df = df.drop(
    columns=[" Bwd PSH Flags", " Bwd URG Flags", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
             " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"], axis=1)

features = df.columns.values[0:-1]

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:, df.columns.values[-1]].values

'''
# Standardize the features
'''

x = StandardScaler().fit_transform(x)

'''
# Train and Test split
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

df_features = pd.DataFrame(x_train, columns=features)
df_values = pd.DataFrame(y_train, columns=[' Label'])


train = pd.concat([df_features, df_values], axis=1)

# Check elements and its count
unique_elelments, count_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elelments, count_elements)))

'''
# Data modeling
'''

class_benign = train[(train[' Label'] == 'BENIGN')]

class_ddos = train[(train[' Label'] == 'DDoS')]

class_dos_goldeneye = train[(train[' Label'] == 'DoS GoldenEye')]
class_dos_hulk = train[(train[' Label'] == 'DoS Hulk')]
class_dos_slowloris = train[(train[' Label'] == 'DoS slowloris')]
class_dos_slowhttptest = train[(train[' Label'] == 'DoS Slowhttptest')]

class_bot = train[(train[' Label'] == 'Bot')]
class_ftp_patator = train[(train[' Label'] == 'FTP-Patator')]
class_heartbleed = train[(train[' Label'] == 'Heartbleed')]
class_infiltration = train[(train[' Label'] == 'Infiltration')]
class_portscan = train[(train[' Label'] == 'PortScan')]
class_ssh_patator = train[(train[' Label'] == 'SSH-Patator')]
class_web_bf = train[(train[' Label'] == 'Web Attack � Brute Force')]
class_web_sql = train[(train[' Label'] == 'Web Attack � Sql Injection')]
class_web_xss = train[(train[' Label'] == 'Web Attack � XSS')]

list_of_base = [class_benign[0:10000], class_bot, class_ftp_patator,
                class_heartbleed, class_infiltration, class_portscan, class_ssh_patator,
                class_web_bf, class_web_sql, class_web_xss]
class0 = pd.concat(list_of_base)

list_of_class1 = [class_benign[10000:20000], class_dos_slowhttptest]
class1 = pd.concat(list_of_class1)

list_of_class2 = [class_benign[20000:30000], class_dos_hulk]
class2 = pd.concat(list_of_class2)

list_of_class3 = [class_benign[30000:40000], class_ddos]
class3 = pd.concat(list_of_class3)

list_of_class4 = [class_benign[40000:50000], class_dos_goldeneye]
class4 = pd.concat(list_of_class4)

list_of_class5 = [class_benign[50000:60000], class_dos_slowloris]
class5 = pd.concat(list_of_class5)


'''
Input C and gamma
'''
c = input("Enter C value: ")
print ("C = "+str(c))
gamma = input("Enter gamma value: ")
print ("gamma = "+str(gamma))


'''

# 1. Standard SVM

'''

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma, decision_function_shape='ovr')

# Initialize values
t_phase = 0

cand_vec_x = []
cand_vec_y = []

x_cur = []
y_cur = []

total_class_seen = []
total_class_seen = np.array(total_class_seen)

fit_training_time_sum = 0

list_of_class_to_train = []

while True:

    print("\n-----------------------------------------------------------")
    print("\nTraining phase " + str(t_phase + 1) + "\n")
    print("\n-----------------------------------------------------------")

    if t_phase == 0:

        list_of_class_to_train = [class0]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 1:

        list_of_class_to_train = [class0, class1]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 2:

        list_of_class_to_train = [class0, class1, class2]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 3:

        list_of_class_to_train = [class0, class1, class2, class3]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 4:

        list_of_class_to_train = [class0, class1, class2, class3, class4]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 5:

        # last iteration
        list_of_class_to_train = [class0, class1, class2, class3, class4, class5]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

        t_phase = -1

    print("# of samples to train in this phase: " + str(len(x_cur)))

    start = time.time()
    svclassifier.fit(x_cur, y_cur)
    end = time.time()

    print("training duration: " + str(end - start))
    fit_training_time_sum += (end - start)
    print("accumulated train time: " + str(fit_training_time_sum))

    start = time.time()
    y_pred = svclassifier.predict(x_test)
    end = time.time()

    print("test duration: " + str(end - start))

    print(classification_report(y_test, y_pred))

    print("Number of Support vectors got in this phase: " + str(len(svclassifier.support_)))

    if t_phase == -1:
        break

    t_phase += 1



'''

# 2. Simple ISVM

'''

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma, decision_function_shape='ovr')

# Initialize values
t_phase = 0

cand_vec_x = []
cand_vec_y = []

x_cur = []
y_cur = []

total_class_seen = []

fit_training_time_sum = 0

while True:

    print("\n-----------------------------------------------------------")
    print("\nTraining phase " + str(t_phase + 1) + "\n")
    print("\n-----------------------------------------------------------")

    if t_phase == 0:

        list_of_class_to_train = [class0]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 1:

        list_of_class_to_train = [class1]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 2:

        list_of_class_to_train = [class2]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 3:

        list_of_class_to_train = [class3]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 4:

        list_of_class_to_train = [class4]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 5:

        # last iteration
        list_of_class_to_train = [class5]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

        t_phase = -1

    unique_elelments, count_elements = np.unique(y_cur, return_counts=True)
    print(np.asarray((unique_elelments, count_elements)))

    # Merge CVs to training set (x_cur, y_cur)
    if t_phase != 0:

        x_cur = np.append(x_cur, cand_vec_x, axis=0)
        y_cur = np.append(y_cur, cand_vec_y, axis=0)

    print("# of samples to train in this phase: " + str(len(x_cur)))

    start = time.time()
    svclassifier.fit(x_cur, y_cur)
    end = time.time()

    print("training duration: " + str(end - start))
    fit_training_time_sum += (end - start)
    print("accumulated train time: " + str(fit_training_time_sum))

    start = time.time()
    y_pred = svclassifier.predict(x_test)
    end = time.time()

    print("test duration: " + str(end - start))

    print(classification_report(y_test, y_pred))

    cand_vec_x = []
    cand_vec_y = []

    print("Number of Support vectors got in this phase: " + str(len(svclassifier.support_)))

    for idx in svclassifier.support_:

        cand_vec_x.append(x_cur[idx])
        cand_vec_y.append(y_cur[idx])

    if t_phase == -1:
        break

    t_phase += 1



'''

# 3. KNN-ISVM

'''

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma, decision_function_shape='ovr')

# Initialize values
t_phase = 0

cand_vec_x = []
cand_vec_y = []

x_cur = []
y_cur = []

total_class_seen = []
total_class_seen = np.array(total_class_seen)

fit_training_time_sum = 0

list_of_class_to_train = []

while True:

    print("\n-----------------------------------------------------------")
    print("\nTraining phase " + str(t_phase + 1) + "\n")
    print("\n-----------------------------------------------------------")

    if t_phase == 0:

        list_of_class_to_train = [class0]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 1:

        list_of_class_to_train = [class1]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 2:

        list_of_class_to_train = [class2]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 3:

        list_of_class_to_train = [class3]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 4:

        list_of_class_to_train = [class4]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

    elif t_phase == 5:

        # last iteration
        list_of_class_to_train = [class5]

        data_train = pd.concat(list_of_class_to_train)

        x_cur = data_train.loc[:, features].values
        y_cur = data_train.loc[:, df.columns.values[-1]].values

        t_phase = -1

    unique_elelments, count_elements = np.unique(y_cur, return_counts=True)
    print(np.asarray((unique_elelments, count_elements)))

    df_features = pd.DataFrame(x_cur, columns=features)
    df_values = pd.DataFrame(y_cur, columns=[' Label'])

    data_train = pd.concat([df_features, df_values], axis=1)

    for element in unique_elelments:

        if element not in total_class_seen:
            total_class_seen = np.append(total_class_seen, element)

    # Merge CVs to training set (x_cur, y_cur)
    if t_phase != 0:

        x_cur = np.append(x_cur, cand_vec_x, axis=0)
        y_cur = np.append(y_cur, cand_vec_y, axis=0)

    print("# of samples to train in this phase: " + str(len(x_cur)))

    start = time.time()
    svclassifier.fit(x_cur, y_cur)
    end = time.time()

    print("training duration: " + str(end - start))
    fit_training_time_sum += (end - start)
    print("accumulated train time: " + str(fit_training_time_sum))

    start = time.time()
    y_pred = svclassifier.predict(x_test)
    end = time.time()

    print("test duration: " + str(end - start))

    print(classification_report(y_test, y_pred))

    '''

    Post processing for selecting candidate vectors

    '''
    # Calculate candidate vectors using knn algorithm
    cand_vec_x = []
    cand_vec_y = []

    neigh = KNeighborsClassifier(n_neighbors=10)

    neigh.fit(x_cur, y_cur)

    # Check class order for class probability array

    unique_elelments, count_elements = np.unique(y_cur, return_counts=True)
    class_seen_this_phase = np.sort(unique_elelments)

    proba_array = neigh.predict_proba(x_cur)

    list_of_idx = []

    for idx in range(len(proba_array)):

        actual_label = y_cur[idx]

        # check index location of the actual label in probability table
        loc = np.where(class_seen_this_phase == actual_label)

        # get the probability of the actual label
        prob = proba_array[idx][loc]

        # enter into candidate vectors

        if prob < 0.7:
            list_of_idx.append(idx)

            cand_vec_x.append(x_cur[idx])
            cand_vec_y.append(y_cur[idx])

    print("Number of Support vectors got in this phase: " + str(len(svclassifier.support_)))

    for idx in svclassifier.support_:

        if idx not in list_of_idx:
            cand_vec_x.append(x_cur[idx])
            cand_vec_y.append(y_cur[idx])

    print("the total number of CSV which will be used for next iteration: " + str(len(cand_vec_x)))

    if t_phase == -1:
        break

    t_phase += 1



