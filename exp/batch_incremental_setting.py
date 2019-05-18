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

# Check elements and its count of the training set
unique_elelments, count_elements = np.unique(y_train, return_counts=True)
print(np.asarray((unique_elelments, count_elements)))


'''

# 1. Standard SVM

'''

print ("*** First model: Standard SVM")

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=1000.0, gamma=0.1, decision_function_shape='ovr')

start = time.time()
svclassifier.fit(x_train, y_train)
end = time.time()

print("training duration: " + str(end - start))

start = time.time()
y_pred = svclassifier.predict(x_test)
end = time.time()

print("test duration: " + str(end - start))

print(classification_report(y_test, y_pred))

print("Number of Support vectors got in this phase: " + str(len(svclassifier.support_)))


'''

# 2. Simple ISVM

'''

print ("*** Second model: Simple ISVM")

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=1000000.0, gamma=0.1, decision_function_shape='ovr')


t_phase = 0
_end = 0

csv_x = []
csv_y = []

x_cur = []
y_cur = []

class_seen = []
class_seen = np.array(class_seen)

training_duration_sum = 0

while True:

    print("\n-----------------------------------------------------------")
    print("\nTraining phase " + str(t_phase) + "\n")

    # Index change
    _st = _end
    _end = _st + 1000

    # Separating out the features
    x_cur = x_train[_st:_end]

    # Separating out the target
    y_cur = y_train[_st:_end]

    if len(x_cur) < 1000:

        # It reached the last increment
        t_phase = -1
        _end = len(x_train)

    if t_phase != 0:

        # Merge CSVs to the training set (x_cur, y_cur)
        x_cur = np.append(x_cur, csv_x, axis=0)
        y_cur = np.append(y_cur, csv_y, axis=0)

    print("# of samples to train in this phase: " + str(len(x_cur)))

    start = time.time()
    svclassifier.fit(x_cur, y_cur)
    end = time.time()

    print("Training duration: " + str(end - start))
    training_duration_sum += (end - start)
    print("Accumulated train time: " + str(training_duration_sum))

    # Print scores every 20 iterations
    if t_phase % 20 == 0:

        start = time.time()
        y_pred = svclassifier.predict(x_test)
        end = time.time()

        print("test duration: " + str(end - start))
        print(classification_report(y_test, y_pred))

    if t_phase == -1: break

    '''
    Post processing for selecting candidate vectors
    '''

    # Calculate candidate vectors for the simple ISVM
    csv_x = []
    csv_y = []

    print("Number of Support vectors got in this phase: "+str(len(svclassifier.support_)))

    for idx in svclassifier.support_:

        csv_x.append(x_cur[idx])
        csv_y.append(y_cur[idx])

    t_phase += 1

'''

# 3. KNN-ISVM

'''

print ("*** Third model: KNN-ISVM")

svclassifier = SVC(kernel='rbf', class_weight='balanced', C=1000.0, gamma=0.1, decision_function_shape='ovr')

t_phase = 0
_end = 0

csv_x = []
csv_y = []

x_cur = []
y_cur = []

class_seen = []
class_seen = np.array(class_seen)

training_duration_sum = 0

while True:

    print("\n-----------------------------------------------------------")
    print("\nTraining phase " + str(t_phase) + "\n")
    print("\n-----------------------------------------------------------")

    # Index change
    _st = _end
    _end = _st + 1000

    # Separating out the features
    x_cur = x_train[_st:_end]

    # Separating out the target
    y_cur = y_train[_st:_end]

    if len(x_cur) < 1000:

        # It reached the last increment
        t_phase = -1
        _end = len(x_train)

    if t_phase != 0:

        # Merge CSVs to training set (x_cur, y_cur)
        x_cur = np.append(x_cur, csv_x, axis=0)
        y_cur = np.append(y_cur, csv_y, axis=0)

    print("# of samples to train in this phase: " + str(len(x_cur)))

    start = time.time()
    svclassifier.fit(x_cur, y_cur)
    end = time.time()

    print("training duration: " + str(end - start))
    training_duration_sum += (end - start)
    print("Accumulated train time: " + str(training_duration_sum))

    # Print scores every 20 iterations
    if t_phase % 20 == 0:

        start = time.time()
        y_pred = svclassifier.predict(x_test)
        end = time.time()

        print("test duration: " + str(end - start))
        print(classification_report(y_test, y_pred))

    if t_phase == -1: break

    '''
    Post processing for selecting candidate vectors
    '''

    # Calculate candidate vectors for the simple ISVM
    csv_x = []
    csv_y = []

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

        if prob < 0.5:

            list_of_idx.append(idx)

            csv_x.append(x_cur[idx])
            csv_y.append(y_cur[idx])

    print ("Number of Support vectors got in this phase: "+str(len(svclassifier.support_)))

    for idx in svclassifier.support_:

        if idx not in list_of_idx:

            csv_x.append(x_cur[idx])
            csv_y.append(y_cur[idx])

    print ("the total number of CSV which will be used for next iteration: " + str(len(csv_x)))

    t_phase += 1










