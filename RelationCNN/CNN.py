"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789


Code was tested with:
- Theano 0.8.2
- Keras 1.1.1
- Python 2.7
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import cPickle as pkl
import gzip
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Merge, GlobalAveragePooling1D
from keras.utils import np_utils
from flipGradientTF import GradientReversal



batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50

print "Load dataset"
f = gzip.open('pkl_tmp/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)


print "sentenceTrain: ", sentenceTrain.shape
print "positionTrain1: ", positionTrain1.shape
print "yTrain: ", yTrain.shape




print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "yTest: ", yTest.shape


f = gzip.open('pkl_tmp/embeddings1.pkl.gz', 'rb')
embeddings1 = pkl.load(f)
f.close()

f = gzip.open('pkl_tmp/embeddings2.pkl.gz', 'rb')
embeddings2 = pkl.load(f)
f.close()

print "Embeddings 1: ",embeddings1.shape
print "Embeddings 2: ",embeddings2.shape


## Position embeddings

distanceModel1 = Sequential()
distanceModel1.add(Embedding(max_position, position_dims, input_length=positionTrain1.shape[1]))

distanceModel2 = Sequential()
distanceModel2.add(Embedding(max_position, position_dims, input_length=positionTrain2.shape[1]))


## Word embeddings

wordModel_dom1 = Sequential()
wordModel_dom1.add(Embedding(embeddings1.shape[0], embeddings1.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings1], trainable=False))

wordModel_dom2 = Sequential()
wordModel_dom2.add(Embedding(embeddings2.shape[0], embeddings2.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings2], trainable=False))


## Domain-agnostic embedding mapper

wordModel_joint = Sequential()
wordModel_joint.add(Merge([wordModel_dom1, wordModel_dom2], mode='concat'))
wordModel_joint.add(Dense(embeddings2.shape[1], activation='tanh'))


## Domain adversarial classifier

domain_classifier = Sequential()
domain_classifier.add(wordModel_joint)
domain_classifier.add(GlobalAveragePooling1D())
domain_classifier.add(GradientReversal(1))
domain_classifier.add(Dense(1))
domain_classifier.add(Activation('sigmoid'))


## Convolutional model pipeline

conv_model = Sequential()
conv_model.add(Merge([wordModel_joint, distanceModel1, distanceModel2], mode='concat'))


conv_model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='tanh',
                        subsample_length=1))
# we use standard max over time pooling
conv_model.add(GlobalMaxPooling1D())

conv_model.add(Dropout(0.25))
conv_model.add(Dense(n_out, activation='softmax'))


## Joint loss

multi_objective_model = Sequential()
multi_objective_model.add(Merge([domain_classifier, conv_model], mode='concat'))


multi_objective_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
multi_objective_model.summary()


print "Start training"



max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in xrange(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

# after each epoch, check error on dev set
for epoch in xrange(nb_epoch):       
    
    sys.stdout.write('[TRAINING] Starting iteration %d...\n' % (epoch+1))
    
    # shuffle training data at start of each epoch
    data_indices = list(range(sentenceTrain.shape[0]))
    np.random.shuffle(data_indices)

    cur_batch_ix, nbatches = 0, 0
    total_batches = np.ceil(sentenceTrain.shape[0]/float(batch_size))
    while cur_batch_ix < sentenceTrain.shape[0]:
        # get the next batch
        next_batch_indices = data_indices[cur_batch_ix:cur_batch_ix+batch_size]
        batch_sentences = sentenceTrain[next_batch_indices]
        batch_labels = train_y_cat[next_batch_indices]
        batch_positions1 = positionTrain1[next_batch_indices]
        batch_positions2 = positionTrain2[next_batch_indices]

        # choose domain
        domain = np.random.choice(
            [1,2],
            p=[0.5, 0.5]
        )
        if domain == 1:
            batch_sentences_1 = batch_sentences
            batch_sentences_2 = np.zeros(batch_sentences.shape)  # "PADDING"
            domain_labels = [[0] for _ in range(batch_size)]
            batch_labels = np.concatenate([batch_labels, domain_labels], axis=1)
        else:
            batch_sentences_1 = np.zeros(batch_sentences.shape)  # "PADDING"
            batch_sentences_2 = batch_sentences
            domain_labels = [[1] for _ in range(batch_size)]
            batch_labels = np.concatenate([batch_labels, domain_labels], axis=1)

        # train on it
        multi_objective_model.train_on_batch(
            [
                batch_sentences_1,
                batch_sentences_2,
                batch_positions1,
                batch_positions2
            ],
            batch_labels
        )

        cur_batch_ix += batch_size
        nbatches += 1

        sys.stdout.write('  >> Processed %d/%d batches\r' % (nbatches, total_batches))
        sys.stdout.flush()

    # ran all batches!
    sys.stdout.write('\n\n[TRAINING] Completed iteration %d.  Calculating dev set error:' % (epoch+1))
    
    #multi_objective_model.fit([sentenceTrain, sentenceTrain, positionTrain1, positionTrain2], train_y_cat, batch_size=batch_size, verbose=True, nb_epoch=1)

    pred_test = multi_objective_model.predict_classes([sentenceTest, sentenceTest, positionTest1, positionTest2], verbose=False)
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(yTest)
   
    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)

    f1Sum = 0
    f1Count = 0
    for targetLabel in xrange(1, max(yTest)):        
        prec = getPrecision(pred_test, yTest, targetLabel)
        rec = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)    
    max_f1 = max(max_f1, macroF1)
    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)
