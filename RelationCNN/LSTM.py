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
min_epoch = 100
max_epoch = 100
convergence_threshold = 0.005
eval_on_test_at_end = False
position_dims = 50
base_lambda = 0.1    # weight of the domain classifier loss
eta = 0.6            # likelihood to use only the domain classifier loss,
                     #   when training on a target domain sample
domain_adaptation = True
pkl_dir = 'pkl_tmp'

#embdim = 300
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

class MyLayer(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_shape = (embdim,embdim)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.W_shape),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape)

def get_R(X):
    U_w, vecT = X[0], X[1]
    print U_w.shape, vecT.shape
    ans = K.batch_dot(U_w, vecT)
    return ans

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
    input_e1 = Input(shape=(1,), name='e1_token')
    input_e2 = Input(shape=(1,), name='e2_token')

    word_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False, input_length=input_shape)
    e_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)

    embedded_words = word_embeddings(input_words)
    embedded_e1 = e_embeddings(input_e1)
    embedded_e2 = e_embeddings(input_e2)

    return ([input_words, input_e1, input_e2], embedded_words, embedded_e1, embedded_e2)

def AddTwoDomainMappedWordEmbeddings(input_shape, mapped_size, embeddings1, embeddings2):
    input_words_dom1 = Input(shape=[input_shape], name='source_word_indices')
    input_words_dom2 = Input(shape=[input_shape], name='target_word_indices')
    input_e1_dom1 = Input(shape=(1,), name='source_e1_token')
    input_e1_dom2 = Input(shape=(1,), name='target_e1_token')
    input_e2_dom1 = Input(shape=(1,), name='source_e2_token')
    input_e2_dom2 = Input(shape=(1,), name='target_e2_token')

    word_embeddings_dom1 = Embedding(embeddings1.shape[0], embeddings1.shape[1], weights=[embeddings1], trainable=False, input_length=sentenceTrain.shape[1], name='source_embeddings')
    word_embeddings_dom2 = Embedding(embeddings2.shape[0], embeddings2.shape[1], weights=[embeddings2], trainable=False, input_length=sentenceTrain.shape[1], name='target_embeddings')
    e_embeddings_dom1 = Embedding(embeddings1.shape[0], embeddings1.shape[1], weights=[embeddings1], trainable=False, name='source_entity_embeddings')
    e_embeddings_dom2 = Embedding(embeddings2.shape[0], embeddings2.shape[1], weights=[embeddings2], trainable=False, name='target_entity_embeddings')

    embedded_words_dom1 = word_embeddings_dom1(input_words_dom1)
    embedded_words_dom2 = word_embeddings_dom2(input_words_dom2)
    embedded_e1_dom1 = e_embeddings_dom1(input_e1_dom1)
    embedded_e1_dom2 = e_embeddings_dom2(input_e1_dom2)
    embedded_e2_dom1 = e_embeddings_dom1(input_e2_dom1)
    embedded_e2_dom2 = e_embeddings_dom2(input_e2_dom2)


    ## Domain-agnostic embedding mapper

    inter_domain_input = Concatenate()([embedded_words_dom1, embedded_words_dom2])
    inter_domain_e1 = Concatenate()([embedded_e1_dom1, embedded_e1_dom2])
    inter_domain_e2 = Concatenate()([embedded_e2_dom1, embedded_e2_dom2])

    domain_mapper = Dense(mapped_size, activation='tanh')

    mapped_input = domain_mapper(inter_domain_input)
    mapped_e1 = domain_mapper(inter_domain_e1)
    mapped_e2 = domain_mapper(inter_domain_e2)

    return ([
        input_words_dom1, input_words_dom2,
        input_e1_dom1, input_e1_dom2,
        input_e2_dom1, input_e2_dom2
    ], mapped_input, mapped_e1, mapped_e2)


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
    embedding_inputs, mapped_words, e1_embed, e2_embed = AddTwoDomainMappedWordEmbeddings(sentlen, mapped_dims, embeddings1, embeddings2)
    embdim = mapped_dims
else:
    embedding_inputs, mapped_words, e1_embed, e2_embed = AddSingleDomainWordEmbeddings(sentlen, embeddings1)

for inp in embedding_inputs: model_inputs.append(inp)


#Input3 = Input(shape=(sentlen,))  # sentence tokens
#Input4 = Input(shape=(1,))    # e1 token
#Input5 = Input(shape=(1,))    # e2 token
#wordModel = Input3
#wordModel = Embedding(embeddings.shape[0], embeddings.shape[1],
#                      input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False)(wordModel)
#e1Model = Embedding(embeddings.shape[0], embeddings.shape[1],
#                      input_length=1, weights=[embeddings], trainable=False)(Input4)
#e2Model = Embedding(embeddings.shape[0], embeddings.shape[1],
#                      input_length=1, weights=[embeddings], trainable=False)(Input5)
#
#
#x_embed = wordModel
#e1_embed = e1Model
e1_embed =  Reshape([embdim,1])(e1_embed)
#e2_embed = e2Model
e2_embed =  Reshape([embdim,1])(e2_embed)

print mapped_words.shape
W1x = MyLayer()(mapped_words)
print W1x.shape
W2x = MyLayer()(mapped_words)
A1 = merge([W1x, e1_embed], output_shape=(sentlen, 1), mode=get_R)
A1 = Lambda(lambda x: x / embdim)(A1)
print A1.shape
A2 = merge([W2x, e2_embed], output_shape=(sentlen, 1), mode=get_R)
A2 = Lambda(lambda x: x / embdim)(A2)
print A2.shape
#alpha1 = Activation("softmax")(A1)
#alpha2 = Activation("softmax")(A2)
alpha1 = Lambda(lambda x: softmax(x,axis=0))(A1)
alpha2 = Lambda(lambda x: softmax(x,axis=0))(A2)
alpha = average([alpha1, alpha2])
alpha = Flatten()(alpha)
alpha = RepeatVector(embdim)(alpha)
alpha = Reshape([sentlen, embdim])(alpha)
print alpha.shape
att_output = multiply([mapped_words, alpha])


c_models = merge([distanceModel1, distanceModel2, POSModel2, att_output], mode='concat', concat_axis=-1)

c_models = Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='tanh',
                        subsample_length=1)(c_models)

c_models = Bidirectional(LSTM(150))(c_models)
c_models = Dropout(0.25)(c_models)
c_models = Dense(n_out, activation='softmax')(c_models)
model_outputs.append(c_models)


## Domain adversarial classifier

if domain_adaptation:
    domain_classifier = AddDomainAdversarialClassifier(mapped_words, use_pooling=True)
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


#main_model = Model(inputs=model_inputs, outputs=c_models)
#main_model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
#main_model.summary()


################################################## Model Complete ####################################################


################################################## Training begins ###################################################

#print "Start training"
#max_prec, max_rec, max_acc, max_f1 = 0,0,0,0
#for epoch in xrange(50):
#    print epoch
#    main_model.fit([positionTrain1, positionTrain2, pos_tags_train, sentenceTrain, e1Train, e2Train], train_y_cat, batch_size=batch_size, verbose=True,nb_epoch=1)
#    pred_vals = main_model.predict([ positionTest1, positionTest2, pos_tags_test, sentenceTest, e1Test, e2Test], verbose=False)
#    pred_test = []
#    for idx in range(len(pred_vals)):
#        pred_test.append(np.argmax(pred_vals[idx]))
#
#    pred_test=np.array(pred_test)
#    dctLabels = np.sum(pred_test)
#    totalDCTLabels = np.sum(yTest)
#
#    acc =  np.sum(pred_test == yTest) / float(len(yTest))
#    max_acc = max(max_acc, acc)
#    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)
#
#    f1Sum = 0
#    f1Count = 0
#    for targetLabel in xrange(0, max(yTest)+1):
#        prec = getPrecision(pred_test, yTest, targetLabel)
#        rec = getPrecision(yTest, pred_test, targetLabel)
#        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
#        #print f1
#        f1Sum += f1
#        f1Count +=1
#
#
#    macroF1 = f1Sum / float(f1Count)
#    max_f1 = max(max_f1, macroF1)
#    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)


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
        batch_postags = pos_tags_train[next_batch_indices]
        batch_e1 = e1Train[next_batch_indices]
        batch_e2 = e2Train[next_batch_indices]

        # choose domain
        domain = np.random.choice(
            [1,2],
            p=[0.5, 0.5]
        )
        # source domain
        if not domain_adaptation or domain == 1:
            batch_sentences_1 = batch_sentences
            batch_sentences_2 = np.zeros(batch_sentences.shape)  # "PADDING"
            batch_e1_1 = batch_e1
            batch_e1_2 = np.zeros(batch_e1.shape)
            batch_e2_1 = batch_e2
            batch_e2_2 = np.zeros(batch_e2.shape)
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
            batch_e1_1 = np.zeros(batch_e1.shape)
            batch_e1_2 = batch_e1
            batch_e2_1 = np.zeros(batch_e2.shape)
            batch_e2_2 = batch_e2
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
            batch_inputs.extend([batch_sentences_1, batch_sentences_2, batch_e1_1, batch_e1_2, batch_e2_1, batch_e2_2])
        else:
            batch_inputs.extend([batch_sentences_1, batch_e1_1, batch_e2_1])
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
            np.zeros(e1Dev.shape),
            e1Dev,
            np.zeros(e2Dev.shape),
            e2Dev,
            np.array([[0] for _ in range(sentenceDev.shape[0])])
        ]
    else:
        inputs = [
            positionDev1, 
            positionDev2,
            pos_tags_dev,
            sentenceDev, 
            e1Dev,
            e2Dev,
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

    # save the best-performing model (ON DEV)
    if macroF1 == max_f1 and eval_on_test_at_end:
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
            np.zeros(e1Test.shape),
            e1Test,
            np.zeros(e2Test.shape),
            e2Test,
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]
    else:
        inputs = [
            positionTest1, 
            positionTest2,
            pos_tags_test,
            sentenceTest, 
            e1Test,
            e2Test,
            np.array([[0] for _ in range(sentenceTest.shape[0])])
        ]

    acc, macroF1 = evaluation(model, inputs, yTest)
    print "Test set accuracy: %.4f" % acc
    print "Non-other Macro-Averaged F1: %.4f\n" % macroF1

################################################## Training terminates ###################################################

model.save("attention.model")
