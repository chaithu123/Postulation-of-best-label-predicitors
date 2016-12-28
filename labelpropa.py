import pandas as pd
import numpy as np
from numpy import *
import operator
import collections
import itertools
labeled_dataset_new=pd.read_csv('C:/Users/chaithu/Desktop/bc_dummy.csv')
pot=len(labeled_dataset_new)
cot=int(pot/3)
cd=[]
for i in range(0,2):
    cd.append(labeled_dataset_new.sample(cot,replace=False)) 
    labeled_dataset_new=labeled_dataset_new.drop(cd[i].index.values)
cd.append(labeled_dataset_new)
cm=int(input("enter the labeled data subset number,either 1 or 2 or 3"))
um=cm-1
if(um==0):
    labeled_dataset=cd[0]
    frames=[cd[1],cd[2]]
    unlabeled_dataset=pd.concat(frames)
    yu=labeled_dataset
    yv=unlabeled_dataset
elif(um==1):
    labeled_dataset=cd[1]
    frames=[cd[0],cd[2]]
    unlabeled_dataset=pd.concat(frames)
    yu=labeled_dataset
    yv=unlabeled_dataset
elif(um==2):
    labeled_dataset=cd[2]
    frames=[cd[0],cd[1]]
    unlabeled_dataset=pd.concat(frames)
    yu=labeled_dataset
    yv=unlabeled_dataset
else:
    print("enter valid labeled data subset number")
col_val=list(labeled_dataset.columns.values)
ncol=len(col_val)
a=len(labeled_dataset)
b=len(unlabeled_dataset)
uu=input("enter the sampling cofficient for the labeled data")
vv=input("enter the sampling cofficient for the unlabeled data")
mun=float(input("enter the threshold"))
mun1=mun*100
mun2=int(mun1)
print(type(mun))
c=int(uu)
d=int(vv)
e=int(a/c)
f=int(b/d)
num=int(input("pls enter max number of iterations"))
k=int(input("enter the internal voting coefficient"))
counter=0
while True:
    counter+=1
    labeled_dataset=yu
    unlabeled_dataset=yv
    unlabeltest=[]
    for i in range(0,d-1):
        unlabeltest.append(unlabeled_dataset.sample(f,replace=False))
        unlabeled_dataset=unlabeled_dataset.drop(unlabeltest[i].index.values)
    unlabeltest.append(unlabeled_dataset)
    unlabel_labels=[]
    for i in range(0,d):
            unlabel_labels.append(unlabeltest[i][col_val[ncol-1]])
            del unlabeltest[i][col_val[ncol-1]]
    labeledtrain=[]
    for i in range(0,c-1):
        labeledtrain.append(labeled_dataset.sample(e,replace=False)) 
        labeled_dataset=labeled_dataset.drop(labeledtrain[i].index.values)
    labeledtrain.append(labeled_dataset)
    label_labels=[]
    for i in range(0,c):
        label_labels.append(labeledtrain[i][col_val[ncol-1]])
        del labeledtrain[i][col_val[ncol-1]]
    def classify0(intX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(intX, (dataSetSize, 1)) - dataSet 
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        p=list(sortedDistIndicies)
        q=list(labels)
        voteIlabel=[]
        for i in range(0,k):
            voteIlabel.append(q[p.index(i)])
        counter=collections.Counter(voteIlabel)
        return (counter.most_common(1)[0][0])   
    s=[]
    py=[]
    for i in range(0,d):
        x=array(unlabeltest[i])
        t=x.__len__()
        for q in range(0,t):
            for p in range(0,c):
                s.append(classify0(x[q],labeledtrain[p],label_labels[p],k))
                if p==c-1:
                    py.append(s)
                    s=[]
    t=[]
    for i in range(0,b):
        counter1=collections.Counter(py[i])
        t.append(counter1.most_common(1)[0][0])
    g=[]
    for i in range(0,d):
        y=list(unlabel_labels[i])
        g.append(y)
    r= list(itertools.chain.from_iterable(g))
    count=0
    for i in range(0,len(r)):
        if t[i]==r[i]:
            count+=1
    accuracy=(count/len(r))*100
    if(counter==num):
        print("it reached max iterations")
        break
    if(accuracy>=mun2):
        print("now u r good to upload ur training dataset")
        break    





import pandas as pd
import numpy as np
from numpy import *
import operator
import collections
import itertools

unlabeled_dataset=pd.read_csv('C:/Users/chaithu/Desktop/bc_unlabel_test.csv')
col_val=list(unlabeled_dataset.columns.values)
ncol=len(col_val)

b=len(unlabeled_dataset)



d=1
f=int(b/d)
unlabeltest=[]
for i in range(0,d-1):
    unlabeltest.append(unlabeled_dataset.sample(f,replace=False))
    unlabeled_dataset=unlabeled_dataset.drop(unlabeltest[i].index.values)
unlabeltest.append(unlabeled_dataset)
unlabel_labels=[]
for i in range(0,d):
    unlabel_labels.append(unlabeltest[i][col_val[ncol-1]])
    del unlabeltest[i][col_val[ncol-1]]   

def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet 
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    p=list(sortedDistIndicies)
    q=list(labels)
    voteIlabel=[]
    for i in range(0,k):
        voteIlabel.append(q[p.index(i)])
    counter=collections.Counter(voteIlabel)
    return (counter.most_common(1)[0][0])
s=[]
py=[]
for i in range(0,d):
    x=array(unlabeltest[i])
    t=x.__len__()
    for q in range(0,t):
        for p in range(0,c):
            s.append(classify0(x[q],labeledtrain[p],label_labels[p],k))
            if p==c-1:
                py.append(s)
                s=[]
t=[]
for i in range(0,b):
    counter1=collections.Counter(py[i])
    t.append(counter1.most_common(1)[0][0])
g=[]
for i in range(0,d):
    y=list(unlabel_labels[i])
    g.append(y)
r= list(itertools.chain.from_iterable(g))
count=0
for i in range(0,len(r)):
    if t[i]==r[i]:
        count+=1
accuracy=(count/len(r))*100
print("labels are predicted at the accuracy of")
print(accuracy)
        
