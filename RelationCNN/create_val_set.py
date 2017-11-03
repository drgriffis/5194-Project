import numpy as np
import os
valid_set,train_set,reduced_train_set= [],[],[]
with open('/users/PAS1315/osu9082/5194-Project/RelationCNN/files/train_ddi.txt') as f:

     for line in f:
         line = line.strip('\n')
         train_set.append(line)

train_set = np.asarray(train_set)
valid_set = np.random.choice(train_set, int(0.15*len(train_set)), replace=False, p=None)

for v in train_set:
    if v not in valid_set:
        reduced_train_set.append(v)


with open('validation_ddi','w') as f:
  for line in valid_set:
    line=line.strip('\n')
    line=line.split('\t')
    for i in range(len(line)):
        if i!=len(line)-1:
           f.write(line[i]+'\t')
        else:
           f.write(line[i]+os.linesep)

with open('reduced_train_ddi','w') as f:
  for line in reduced_train_set:
    line=line.strip('\n')
    line=line.split('\t')
    for i in range(len(line)):
        if i!=len(line)-1:
           f.write(line[i]+'\t')
        else:
           f.write(line[i]+os.linesep)
