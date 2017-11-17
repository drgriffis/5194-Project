import numpy as np
import os
import sys
def k_fold_cross_validation(X, K, randomise = False):
	"""
	Generates K (training, validation) pairs from the items in X.

	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	for k in xrange(K):
		training = [x for i, x in enumerate(X) if i % K != k]
		validation = [x for i, x in enumerate(X) if i % K == k]
		yield training, validation

def write_info(reduced_train_set,valid_set,dataset,idx):


 with open('newData/validation_'+dataset+'_'+str(idx),'w') as f:

  for line in valid_set:
    line=line.strip('\n')
    line=line.split('\t')
    for i in range(len(line)):
        if i!=len(line)-1:
           f.write(line[i]+'\t')
        else:
           f.write(line[i]+os.linesep)

 with open('newData/reduced_train_'+dataset+'_'+str(idx),'w') as f:

  for line in reduced_train_set:
    line=line.strip('\n')
    line=line.split('\t')
    for i in range(len(line)):
        if i!=len(line)-1:
           f.write(line[i]+'\t')
        else:
           f.write(line[i]+os.linesep)
valid_set,train_set,reduced_train_set= [],[],[]
dataset = sys.argv[1]
with open('/users/PAS1315/osu9082/5194-Project/RelationCNN/files/'+dataset) as f:
     for line in f:
         line = line.strip('\n')
         train_set.append(line)

train_set = np.asarray(train_set)
someSet = list(k_fold_cross_validation(train_set,5,True))
i=0
for x,y in someSet:
    print (i)
    write_info(x,y,dataset,i)
    i=i+1

