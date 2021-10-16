#-------------------------------------------------------------------------
# AUTHOR: Sara Nersisian
# FILENAME: decision_tree.py
# SPECIFICATION: decision tree program A1.7d
# FOR: CS 4210.01- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
X = []

for row in range(len(db)):
    X_row=[]
    for col in range(len(db[row])-1):
        if db[row][col] == "Young":
            X_row.append(1)
        elif db[row][col] == "Presbyopic":
            X_row.append(2)
        elif db[row][col] == "Prepresbyopic":
            X_row.append(3)
        elif db[row][col] == "Myope":
            X_row.append(1)
        elif db[row][col] == "Hypermetrope":
            X_row.append(2)
        elif db[row][col] == "Yes":
            X_row.append(1)
        elif db[row][col] == "No":
            X_row.append(2)
        elif db[row][col] == "Reduced":
            X_row.append(1)
        elif db[row][col] == "Normal":
            X_row.append(2)
    X.append(X_row)

print("X =",X)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for row in range(len(db)):
    if db[row][4]== "Yes":
        Y.append(1)
    elif db[row][4] == "No":
        Y.append(2)

print("Y =", Y)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()