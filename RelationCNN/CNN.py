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
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Merge, GlobalAveragePooling1D
from keras.layers.merge import Concatenate, Add
from keras.utils import np_utils
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras import backend as K
from flipGradientTF import GradientReversal


def AddDomainAdversarialClassifier(mapped_input, use_pooling=False):
    """Takes a Keras object containing embeddings that have been mapped
    to a common representation, and returns a binary domain classifier
    with gradient reversal.
    """
    if use_pooling:
        domain_pooled_input = GlobalAveragePooling1D()(mapped_input)
        grad_rev = GradientReversal(1)(domain_pooled_input)
    else:
        grad_rev = GradientReversal(1)(mapped_input)
    domain_transformer = Dense(1)(grad_rev)
    domain_classifier = Activation('sigmoid')(domain_transformer)

    return domain_classifier


def AddSingleDomainWordEmbeddings(input_shape, embeddings):
    input_words = Input(shape=[input_shape], name='word_indices')
    word_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False, input_length=input_shape)
    embedded_words = word_embeddings(input_words)

    return ([input_words], embedded_words)

def AddTwoDomainMappedWordEmbeddings(input_shape, mapped_size, embeddings1, embeddings2):
    ## Word embeddings

    input_words_dom1 = Input(shape=[input_shape], name='dom1_word_indices')
    input_words_dom2 = Input(shape=[input_shape], name='dom2_word_indices')

    word_embeddings_dom1 = Embedding(embeddings1.shape[0], embeddings1.shape[1], weights=[embeddings1], trainable=False, input_length=sentenceTrain.shape[1])
    word_embeddings_dom2 = Embedding(embeddings2.shape[0], embeddings2.shape[1], weights=[embeddings2], trainable=False, input_length=sentenceTrain.shape[1])

    embedded_words_dom1 = word_embeddings_dom1(input_words_dom1)
    embedded_words_dom2 = word_embeddings_dom2(input_words_dom2)


    ## Domain-agnostic embedding mapper

    inter_domain_input = Concatenate()([embedded_words_dom1, embedded_words_dom2])
    domain_mapper = Dense(mapped_size, activation='tanh')
    mapped_input = domain_mapper(inter_domain_input)

    return ([input_words_dom1, input_words_dom2], mapped_input)


batch_size = 64
filters = 100
kernel_size = 3
hidden_dims = 100
mapped_dims = 100
min_epoch = 100
max_epoch = 100
convergence_threshold = 0.005
eval_on_test_at_end = False
position_dims = 50
base_lambda = 0.4    # weight of the domain classifier loss
eta = 0.3            # likelihood to use only the domain classifier loss,
                     #   when training on a target domain sample
#domain_adaptation = True
#pkl_dir = 'pkl_tmp'
#domain_adaptation = False
#pkl_dir = 'pkl_tmp_single_vocab'
domain_adaptation = True
pkl_dir = 'pkl_ddi_pubmed_gigaword'
#pkl_dir = 'pkl_ddi_gigaword_pubmed'
#pkl_dir = 'pkl_semeval_gigaword_wikipedia'


if len(sys.argv) == 3:
    base_lambda = float(sys.argv[1])
    eta = float(sys.argv[2])
    print '--- CLI OVERRIDES ---'
    print '  base_lambda: %0.2f' % base_lambda
    print '  eta: %0.2f' % eta
    print

print "Load dataset"
f = gzip.open('%s/sem-relations.pkl.gz' % pkl_dir, 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yDev, sentenceDev, positionDev1, positionDev2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)


print "sentenceTrain: ", sentenceTrain.shape
print "positionTrain1: ", positionTrain1.shape
print "yTrain: ", yTrain.shape

print "sentenceDev: ", sentenceDev.shape
print "positionDev1: ", positionDev1.shape
print "yDev: ", yDev.shape

print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "yTest: ", yTest.shape


if domain_adaptation:
    f = gzip.open('%s/embeddings1.pkl.gz' % pkl_dir, 'rb')
    embeddings1 = pkl.load(f)
    f.close()
    print "Embeddings 1: ",embeddings1.shape

    f = gzip.open('%s/embeddings2.pkl.gz' % pkl_dir, 'rb')
    embeddings2 = pkl.load(f)
    f.close()
    print "Embeddings 2: ",embeddings2.shape
else:
    f = gzip.open('%s/embeddings1.pkl.gz' % pkl_dir, 'rb')
    embeddings1 = pkl.load(f)
    f.close()
    print "Embeddings: ",embeddings1.shape



model_inputs = []
model_outputs = []
model_losses = []


## Position embeddings

input_positions_e1 = Input(shape=[positionTrain1.shape[1]], name='e1_positions')
input_positions_e2 = Input(shape=[positionTrain2.shape[1]], name='e2_positions')

position_embeddings_e1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])
position_embeddings_e2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])

embedded_positions_e1 = position_embeddings_e1(input_positions_e1)
embedded_positions_e2 = position_embeddings_e2(input_positions_e2)

model_inputs.append(input_positions_e1)
model_inputs.append(input_positions_e2)


## Word embeddings

if domain_adaptation:
    embedding_inputs, mapped_input = AddTwoDomainMappedWordEmbeddings(sentenceTrain.shape[1], mapped_dims, embeddings1, embeddings2)
else:
    embedding_inputs, mapped_input = AddSingleDomainWordEmbeddings(sentenceTrain.shape[1], embeddings1)

for inp in embedding_inputs: model_inputs.append(inp)

## Convolutional model pipeline

conv_input = Concatenate()([mapped_input, embedded_positions_e1, embedded_positions_e2])


convolution = Conv1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='tanh',
                        strides=1)
convolved = convolution(conv_input)
# we use standard max over time pooling
pooled = GlobalMaxPooling1D()(convolved)

dropout = Dropout(0.25)(pooled)
conv_prediction = Dense(n_out, activation='softmax')(dropout)
model_outputs.append(conv_prediction)


## Domain adversarial classifier

if domain_adaptation:
    domain_classifier = AddDomainAdversarialClassifier(mapped_input, use_pooling=True)
    model_outputs.append(domain_classifier)


## Per-task loss functions

tradeoff_param = Input(shape=(1,), name='tradeoff_param')
model_inputs.append(tradeoff_param)

def main_task_loss(y_pred, y_true):
    raw_loss = categorical_crossentropy(y_pred, y_true)
    return (1-tradeoff_param)*raw_loss

def domain_classifier_loss(y_pred, y_true):
    raw_loss = binary_crossentropy(y_pred, y_true)
    return tradeoff_param*raw_loss

if domain_adaptation: model_losses = [main_task_loss, domain_classifier_loss]
else: model_losses = [main_task_loss]

model = Model(
    inputs=model_inputs,
    outputs=model_outputs
)
model.compile(
    loss=model_losses,
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

def evaluation(model, inputs, labels):
    if domain_adaptation:
        (soft_predictions, _) = model.predict(inputs, verbose=False)
    else:
        soft_predictions = model.predict(inputs, verbose=False)

    predictions = np.argmax(soft_predictions, axis=1)
    
    dctLabels = np.sum(predictions)
    totalDCTLabels = np.sum(labels)
   
    acc =  np.sum(predictions == labels) / float(len(labels))

    f1Sum = 0
    f1Count = 0
    for targetLabel in xrange(1, max(labels)):        
        prec = getPrecision(predictions, labels, targetLabel)
        rec = getPrecision(labels, predictions, targetLabel)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        f1Sum += f1
        f1Count +=1    
        
    macroF1 = f1Sum / float(f1Count)    

    return (acc, macroF1)

# train until F1 converges (or starts decreasing) on dev set
epoch = 0
prev_f1, f1_increase = 0, float('inf')
while (epoch < min_epoch or f1_increase > convergence_threshold) and (epoch < max_epoch):
    
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
        if not domain_adaptation or domain == 1:
            batch_sentences_1 = batch_sentences
            batch_sentences_2 = np.zeros(batch_sentences.shape)  # "PADDING"
            domain_labels = np.array([[0] for _ in range(batch_sentences.shape[0])])
            if domain_adaptation:
                lmbda = base_lambda
            else:
                lmbda = 0
        # target domain
        elif domain_adaptation:
            batch_sentences_1 = np.zeros(batch_sentences.shape)  # "PADDING"
            batch_sentences_2 = batch_sentences
            domain_labels = np.array([[1] for _ in range(batch_sentences.shape[0])])
            # choose if we will use the joint loss or domain loss only
            loss_type = np.random.choice(
                [1,2],
                p=[1-eta,eta]
            )
            if loss_type == 1:
                lmbda = base_lambda
            else:
                lmbda = 1

        lmbda = np.array([[lmbda] for _ in range(batch_sentences.shape[0])])

        # set up the batch inputs
        batch_inputs = [batch_positions1, batch_positions2]
        if domain_adaptation:
            batch_inputs.extend([batch_sentences_1, batch_sentences_2])
        else:
            batch_inputs.append(batch_sentences_1)
        batch_inputs.append(lmbda)
        # and the batch labels
        if domain_adaptation:
            batch_label_array = [batch_labels, domain_labels]
        else:
            batch_label_array = [batch_labels]

        # train on it
        model.train_on_batch(
            batch_inputs,
            batch_label_array
        )

        cur_batch_ix += batch_size
        nbatches += 1

        sys.stdout.write('  >> Processed %d/%d batches\r' % (nbatches, total_batches))
        sys.stdout.flush()

    # ran all batches!
    sys.stdout.write('\n\n[TRAINING] Completed iteration %d.  Calculating dev set error:\n' % (epoch+1))

    if domain_adaptation:
        inputs = [
            positionDev1, 
            positionDev2,
            sentenceDev, 
            sentenceDev, 
            np.array([[0] for _ in range(sentenceDev.shape[0])])
        ]
    else:
        inputs = [
            positionDev1, 
            positionDev2,
            sentenceDev, 
            np.array([[0] for _ in range(sentenceDev.shape[0])])
        ]

    acc, macroF1 = evaluation(model, inputs, yDev)

    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)

    max_f1 = max(max_f1, macroF1)
    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)

    f1_increase = macroF1 - prev_f1
    prev_f1 = macroF1

    epoch += 1

print('>>> Training complete in %d iterations! <<<' % epoch)

## If specified, evaluate on the test set at the end
if eval_on_test_at_end:
    sys.stdout.write('\n\n[TESTING] Evaluating trained model on test set.\n')

    if domain_adaptation:
        inputs = [
            positionTest1, 
            positionTest2,
            sentenceTest, 
            sentenceTest, 
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]
    else:
        inputs = [
            positionTest1, 
            positionTest2,
            sentenceTest, 
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]

    acc, macroF1 = evaluation(model, inputs, yTest)
    print "Test set accuracy: %.4f" % acc
    print "Non-other Macro-Averaged F1: %.4f\n" % macroF1
