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
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Merge, GlobalAveragePooling1D
from keras.layers.merge import Concatenate, Add
from keras.utils import np_utils
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras import backend as K
from flipGradientTF import GradientReversal



batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50
base_lambda = 0.4

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

#f = gzip.open('pkl_tmp/embeddings2.pkl.gz', 'rb')
f = gzip.open('pkl_tmp/embeddings1.pkl.gz', 'rb')
embeddings2 = pkl.load(f)
f.close()

print "Embeddings 1: ",embeddings1.shape
print "Embeddings 2: ",embeddings2.shape


## Position embeddings

input_positions_e1 = Input(shape=[positionTrain1.shape[1]], name='e1_positions')
input_positions_e2 = Input(shape=[positionTrain2.shape[1]], name='e2_positions')

position_embeddings_e1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])
position_embeddings_e2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])

embedded_positions_e1 = position_embeddings_e1(input_positions_e1)
embedded_positions_e2 = position_embeddings_e2(input_positions_e2)


## Word embeddings

input_words_dom1 = Input(shape=[sentenceTrain.shape[1]], name='dom1_word_indices')
input_words_dom2 = Input(shape=[sentenceTrain.shape[1]], name='dom2_word_indices')

word_embeddings_dom1 = Embedding(embeddings1.shape[0], embeddings1.shape[1], weights=[embeddings1], trainable=False, input_length=sentenceTrain.shape[1])
word_embeddings_dom2 = Embedding(embeddings2.shape[0], embeddings2.shape[1], weights=[embeddings2], trainable=False, input_length=sentenceTrain.shape[1])

embedded_words_dom1 = word_embeddings_dom1(input_words_dom1)
embedded_words_dom2 = word_embeddings_dom2(input_words_dom2)


## Domain-agnostic embedding mapper

inter_domain_input = Add()([embedded_words_dom1, embedded_words_dom2])
domain_mapper = Dense(embeddings2.shape[1], activation='tanh')
mapped_input = domain_mapper(inter_domain_input)


## Domain adversarial classifier

domain_pooled_input = GlobalAveragePooling1D()(mapped_input)
grad_rev = GradientReversal(1)(domain_pooled_input)
domain_transformer = Dense(1)(grad_rev)
domain_classifier = Activation('sigmoid')(domain_transformer)


## Convolutional model pipeline

conv_input = Concatenate()([mapped_input, embedded_positions_e1, embedded_positions_e2])


convolution = Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='tanh',
                        subsample_length=1)
convolved = convolution(conv_input)
# we use standard max over time pooling
pooled = GlobalMaxPooling1D()(convolved)

dropout = Dropout(0.25)(pooled)
conv_prediction = Dense(n_out, activation='softmax')(dropout)


## Per-task loss functions

tradeoff_param = Input(shape=(1,), name='tradeoff_param')

def main_task_loss(y_pred, y_true):
    raw_loss = categorical_crossentropy(y_pred, y_true)
    return (1-tradeoff_param)*raw_loss

def domain_classifier_loss(y_pred, y_true):
    raw_loss = binary_crossentropy(y_pred, y_true)
    return tradeoff_param*raw_loss

multi_objective_model = Model(
    inputs=[input_words_dom1, input_words_dom2, input_positions_e1, input_positions_e2, tradeoff_param],
    outputs=[conv_prediction, domain_classifier]
)
multi_objective_model.compile(
    #loss=joint_loss,
    loss=[main_task_loss, domain_classifier_loss],
    optimizer='Adam',
    metrics=['accuracy']
)


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
        # source domain
        if domain == 1:
            batch_sentences_1 = batch_sentences
            batch_sentences_2 = np.zeros(batch_sentences.shape)  # "PADDING"
            domain_labels = np.array([[0] for _ in range(batch_size)])
            lmbda = base_lambda
        # target domain
        else:
            batch_sentences_1 = np.zeros(batch_sentences.shape)  # "PADDING"
            batch_sentences_2 = batch_sentences
            domain_labels = np.array([[1] for _ in range(batch_size)])
            # choose if we will use the joint loss or domain loss only
            loss_type = np.random.choice(
                [1,2],
                p=[1.0,0.0]
            )
            if loss_type == 1:
                lmbda = base_lambda
            else:
                lmbda = 1

        lmbda = np.array([[lmbda] for _ in range(batch_size)])

        # train on it
        multi_objective_model.train_on_batch(
            [
                batch_sentences_1,
                batch_sentences_2,
                batch_positions1,
                batch_positions2,
                lmbda
            ],
            [
                batch_labels,
                domain_labels
            ]
        )

        cur_batch_ix += batch_size
        nbatches += 1

        sys.stdout.write('  >> Processed %d/%d batches\r' % (nbatches, total_batches))
        sys.stdout.flush()

    # ran all batches!
    sys.stdout.write('\n\n[TRAINING] Completed iteration %d.  Calculating dev set error:' % (epoch+1))

    (soft_predictions, _) = multi_objective_model.predict(
        [
            sentenceTest, 
            sentenceTest, 
            positionTest1, 
            positionTest2,
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ], verbose=False
    )
    pred_test = np.argmax(soft_predictions, axis=1)
    
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
