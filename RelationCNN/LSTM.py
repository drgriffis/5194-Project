import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import cPickle as pkl
import gzip
import keras

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input, Embedding, LSTM, Dense, merge, TimeDistributed, Lambda, average, multiply, Dot, dot
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, Add
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.activations import softmax
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
import keras.backend.tensorflow_backend as K
from keras.regularizers import l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import * 
from flipGradientTF import GradientReversal

import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip


################################################## Loading Data ########################################################

batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
mapped_dims = 100
min_epoch = 50
max_epoch = 50
convergence_threshold = 0.005
eval_on_test_at_end = False
position_dims = 50
base_lambda = 0.1    # weight of the domain classifier loss
eta = 0.6            # likelihood to use only the domain classifier loss,
                     #   when training on a target domain sample
domain_adaptation = True
pkl_dir = 'pkl_tmp'

embdim = 100
sentlen = 73
poslen = 31


modelf = '%s/best_hp_model.h5' % pkl_dir
best_iter = 0

def _cli():
    global pkl_dir, base_lambda, eta
    import optparse
    parser = optparse.OptionParser(usage='Usage: %prog')
    parser.add_option('--datadir', dest='datadir',
        help='directory with pickle files',
        default=pkl_dir)
    parser.add_option('--lambda', dest='lmbda',
        type='float', default=base_lambda,
        help='multitask weight (0=main task only, 1=domain classifier only; default: %default)')
    parser.add_option('--eta', dest='eta',
        type='float', default=eta,
        help='DANN-only sampling rate (0=always joint loss, 1=always DANN only (for target domain); default: %default)')
    parser.add_option('--no-adapt', dest='noadapt',
        action='store_true', default=False,
        help='Disable domain adaptation (uses embeddings1 as the only embeddings)')
    parser.add_option('--test', dest='test',
        action='store_true', default=eval_on_test_at_end,
        help='Enable evaluation on test set at end of training')
    parser.add_option('--emb-dim', dest='embdim',
        type='int', default=100)
    (options, args) = parser.parse_args()

    if options.lmbda != base_lambda:
        print '[CLI OVERRIDE] lambda = %.4f' % options.lmbda
    if options.eta != eta:
        print '[CLI OVERRIDE] eta = %.4f' % options.eta
    if options.embdim != embdim:
        print '[CLI OVERRIDE] embdim = %d' % options.embdim
    return options.datadir, options.lmbda, options.eta, options.embdim, options.noadapt, options.test

pkl_dir, base_lambda, eta, embdim, noadapt, eval_on_test_at_end = _cli()
domain_adaptation = not noadapt

print "Loading dataset from %s" % pkl_dir
f = gzip.open('%s/sem-relations.pkl.gz' % pkl_dir, 'rb')
first_array = pkl.load(f)

if type(first_array) is tuple or first_array.shape != (1,):
    yTrain, sentenceTrain, pos_tags_train, positionTrain1, positionTrain2, e1Train, e2Train = first_array
    nfiles = 3
    print ' >> Assuming train/dev/test'
else:
    nfiles = first_array[0]
    if nfiles == 2: print ' >> Using train/dev'
    elif nfiles == 3: print ' >> Using train/dev/test'
    yTrain, sentenceTrain, pos_tags_train, positionTrain1, positionTrain2, e1Train, e2Train = pkl.load(f)

yDev, sentenceDev, pos_tags_dev, positionDev1, positionDev2, e1Dev, e2Dev = pkl.load(f)

if nfiles > 2:
    yTest, sentenceTest, pos_tags_test, positionTest1, positionTest2, e1Test, e2Test  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)


print "sentenceTrain: ", sentenceTrain.shape
print "positionTrain1: ", positionTrain1.shape
print "e1Train: ", e1Train.shape
print "e2Train: ", e2Train.shape
print "yTrain: ", yTrain.shape

print "sentenceDev: ", sentenceDev.shape
print "positionDev1: ", positionDev1.shape
print "e1Dev: ", e1Dev.shape
print "e2Dev: ", e2Dev.shape
print "yDev: ", yDev.shape

if nfiles > 2:
    print "sentenceTest: ", sentenceTest.shape
    print "positionTest1: ", positionTest1.shape
    print "e1Test: ", e1Test.shape
    print "e2Test: ", e2Test.shape
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

########################################### Data Loading finished ######################################################


######################## MODEL with POS and trainable attention and Bidirectional LSTM

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        #ait = K.dot(uit, self.u)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number \epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        #return K.sum(weighted_input, axis=1)
        print "here", weighted_input.shape
        return weighted_input
        
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

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
    input_words_dom1 = Input(shape=[input_shape], name='source_word_indices')
    input_words_dom2 = Input(shape=[input_shape], name='target_word_indices')

    word_embeddings_dom1 = Embedding(embeddings1.shape[0], embeddings1.shape[1], weights=[embeddings1], trainable=False, input_length=sentenceTrain.shape[1], name='source_embeddings')
    word_embeddings_dom2 = Embedding(embeddings2.shape[0], embeddings2.shape[1], weights=[embeddings2], trainable=False, input_length=sentenceTrain.shape[1], name='target_embeddings')

    embedded_words_dom1 = word_embeddings_dom1(input_words_dom1)
    embedded_words_dom2 = word_embeddings_dom2(input_words_dom2)

    ## Domain-agnostic embedding mapper

    inter_domain_input = Concatenate()([embedded_words_dom1, embedded_words_dom2])
    domain_mapper = Dense(mapped_size, activation='tanh')
    mapped_input = domain_mapper(inter_domain_input)

    return ([input_words_dom1, input_words_dom2], mapped_input)


model_inputs = []
model_outputs = []
model_losses = []


## Domain-invariant inputs 

inp_position_e1 = Input(shape=(sentlen,))  # distance from entity 1
distanceModel1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(inp_position_e1)

inp_position_e2 = Input(shape=(sentlen,))  # distance from entity 2
distanceModel2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])(inp_position_e2)

inp_postags = Input(shape=(sentlen,))  # POS tags for sentence
POSModel2 = Embedding(poslen, position_dims, input_length=pos_tags_train.shape[1])(inp_postags)

model_inputs.append(inp_position_e1)
model_inputs.append(inp_position_e2)
model_inputs.append(inp_postags)


## Word embedding (domain-sensitive) inputs

if domain_adaptation:
    embedding_inputs, wordModel = AddTwoDomainMappedWordEmbeddings(sentlen, mapped_dims, embeddings1, embeddings2)
    embdim = mapped_dims
else:
    embedding_inputs, wordModel = AddSingleDomainWordEmbeddings(sentlen, embeddings1)

for inp in embedding_inputs: model_inputs.append(inp)

wordModel = Bidirectional(LSTM(150, return_sequences = True))(wordModel)
wordModel = AttentionWithContext()(wordModel)
c_models = Concatenate(axis = -1)([distanceModel1, distanceModel2, POSModel2, wordModel])
c_models = Bidirectional(LSTM(150))(c_models)
c_models = Dropout(0.25)(c_models)
c_models = Dense(n_out, activation='softmax')(c_models)
model_outputs.append(c_models)


## Domain adversarial classifier

if domain_adaptation:
    domain_classifier = AddDomainAdversarialClassifier(wordModel, use_pooling=True)
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
    #optimizer='Adam',
    optimizer='rmsprop',
    metrics=['accuracy']
)


################################################## Model Complete ####################################################


################################################## Training begins ###################################################

print "Start training"

max_prec, max_rec, max_acc, max_f1_4c, max_f1_5c = 0,0,0,0,0

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

    f1s = []
    for start in [0,1]:
        f1Sum = 0
        f1Count = 0
        for targetLabel in xrange(start, max(labels)):        
            prec = getPrecision(predictions, labels, targetLabel)
            rec = getPrecision(labels, predictions, targetLabel)
            f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
            f1Sum += f1
            f1Count +=1    
        
        macroF1 = f1Sum / float(f1Count)    
        f1s.append(macroF1)

    return (acc, f1s[0], f1s[1])

# train until F1 converges (or starts decreasing) on dev set
epoch = 0
prev_f1_4c, f1_increase = 0, float('inf')
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
        batch_postags = pos_tags_train[next_batch_indices]

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
                # multi-objective
                lmbda = base_lambda
            else:
                # main objective only
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
            # using joint loss
            if loss_type == 1:
                lmbda = base_lambda
            # using domain loss only
            else:
                lmbda = 1

        lmbda = np.array([[lmbda] for _ in range(batch_sentences.shape[0])])

        # set up the batch inputs
        batch_inputs = [batch_positions1, batch_positions2, batch_postags]
        if domain_adaptation:
            batch_inputs.extend([batch_sentences_1, batch_sentences_2])
        else:
            batch_inputs.extend([batch_sentences_1])
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
            pos_tags_dev,
            np.zeros(sentenceDev.shape), # not using source domain
            sentenceDev,                 # testing from target domain
            np.array([[0] for _ in range(sentenceDev.shape[0])])
        ]
    else:
        inputs = [
            positionDev1, 
            positionDev2,
            pos_tags_dev,
            sentenceDev, 
            np.array([[0] for _ in range(sentenceDev.shape[0])])
        ]

    acc, macroF1_5c, macroF1_4c = evaluation(model, inputs, yDev)

    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)

    max_f1_5c = max(max_f1_5c, macroF1_5c)
    print "5-class Macro-Averaged F1: %.4f (max: %.4f)" % (macroF1_5c, max_f1_5c)
    max_f1_4c = max(max_f1_4c, macroF1_4c)
    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1_4c, max_f1_4c)

    f1_increase = macroF1_4c - prev_f1_4c
    prev_f1_4c= macroF1_4c

    epoch += 1

    # save the best-performing model (ON DEV)
    if macroF1_4c == max_f1_4c and eval_on_test_at_end:
        print '   >> SAVING MODEL WEIGHTS'
        model.save_weights(modelf)
        best_iter = epoch

print('>>> Training complete in %d iterations! <<<' % epoch)
print('>>> Best model performance on dev at epoch %d <<<' % best_iter)

## If specified, evaluate on the test set at the end
if eval_on_test_at_end:
    sys.stdout.write('\n\n[TESTING] Evaluating trained model on test set.\n')
    print('>>> Loading model from epoch %d <<<' % best_iter)
    model.load_weights(modelf)

    if domain_adaptation:
        inputs = [
            positionTest1, 
            positionTest2,
            pos_tags_test,
            np.zeros(sentenceTest.shape), # not using source domain
            sentenceTest,                 # testing from target domain
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]
    else:
        inputs = [
            positionTest1, 
            positionTest2,
            pos_tags_test,
            sentenceTest, 
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]

    acc, macroF1_5c, macroF1_4c = evaluation(model, inputs, yTest)
    print "Test set accuracy: %.4f" % acc
    print "5-class Macro-Averaged F1: %.4f" % macroF1_5c
    print "Non-other Macro-Averaged F1: %.4f\n" % macroF1_4c

################################################## Training terminates ###################################################

model.save("attention.model")
