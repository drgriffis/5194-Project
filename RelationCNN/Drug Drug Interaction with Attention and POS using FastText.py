import numpy as np
np.random.seed(1337)  # for reproducibility

import cPickle as pkl
import gzip
import keras

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input, Embedding, LSTM, Dense, merge, TimeDistributed, Lambda, average, multiply, Dot, dot
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
import keras.backend.tensorflow_backend as K
from keras.regularizers import l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import * 

from IPython.display import SVG

import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip


############################################# Preprocessing Starts ###############################################


def createMatrices(file, word2Idx, pos_tags_dict, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    e1Matrix = []
    e2Matrix = []
    tokenMatrix = []
    pos_tags_Matrix = []
    count = 0
    
    line_no = 0 
    for line in open(file):
        line_no += 1
        splits = line.strip().split('\t')
        
        label = splits[0]
        if "false" == label.lower():
            continue
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens_postags = sentence.split(" ")
        
        tokens = []
        pos_tags = []
        
        for idx in range(len(tokens_postags)):
            if idx % 2 == 0:
                tokens.append(tokens_postags[idx])
            else:
                pos_tags.append(tokens_postags[idx])
                if tokens_postags[idx] not in pos_tags_dict:
                    pos_tags_dict[tokens_postags[idx]] = count
                    count += 1
        
        if len(tokens)!=len(pos_tags):
            print "false", line, line_no
        labelsDistribution[label] += 1
        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        pos_tags_Ids = np.zeros(maxSentenceLen)
        e1 = 0
        e2 = 0
        
        for idx in xrange(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            pos_tags_Ids[idx] = pos_tags_dict[pos_tags[idx]]
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 == 0:
                e1 = tokenIds[idx]
            elif distance2 == 0:
                e2 = tokenIds[idx]
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']
         
        pos_tags_Matrix.append(pos_tags_Ids)
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        e1Matrix.append(e1)
        e2Matrix.append(e2)
        labels.append(labelsMapping[label])
        

    print pos_tags_dict
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(pos_tags_Matrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'), np.array(e1Matrix, dtype='int32'), np.array(e2Matrix, dtype='int32')


def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]


outputFilePath = 'pkl/sem-relations.pkl.gz'
embeddingsPklPath = 'pkl/embeddings.pkl.gz'

embeddingsPath = 'wiki.en.vec'

folder = 'files/'
files = [folder + 'train_ddi_new.txt', folder + 'test_ddi_new.txt']

#Mapping of the labels to integers

labelsMapping = {'false' : 0, 
                 'effect' : 1, 
                 'advise' : 2, 
                 'mechanism' : 3,
                 'int' : 4}

words = {}
pos_tags_dict = {}
maxSentenceLen = [0,0,0]
labelsDistribution = FreqDist()

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in xrange(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)


for fileIdx in xrange(len(files)):
    file = files[fileIdx]
    print files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        
        
        sentence = splits[3]  
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True
    print len(words)

print "Max Sentence Lengths: ", maxSentenceLen, len(words)
        
# :: Read in word embeddings ::
word2Idx = {}
embeddings = []
count = 0
for line in open(embeddingsPath):
    split = line.strip().split(" ")
    word = split[0]
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING"] = len(word2Idx)
        vector = np.zeros(300)
        embeddings.append(vector)
        
        word2Idx["UNKNOWN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, 300)
        embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])
        embeddings.append(vector)
        word2Idx[word.lower()] = len(word2Idx)
        count += 1
        
embeddings = np.array(embeddings)

print "Embeddings shape: ", embeddings.shape
print "Len words: ", len(words), count

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()

# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, pos_tags_dict, max(maxSentenceLen))
print len(train_set)
test_set = createMatrices(files[1], word2Idx, pos_tags_dict, max(maxSentenceLen))
print len(test_set)

f = gzip.open(outputFilePath, 'wb')
pkl.dump(train_set, f, -1)
pkl.dump(test_set, f, -1)
f.close()

print "Data stored in pkl folder"

for label, freq in labelsDistribution.most_common(100):
    print "%s : %f%%" % (label, 100*freq / float(labelsDistribution.N()))
    
    
################################################## Preprocessing Ends ###################################################



################################################## Loading Data ########################################################

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


batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 50
position_dims = 50

print "Load dataset"
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, pos_tags_train, positionTrain1, positionTrain2, e1Train, e2Train = pkl.load(f)
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

print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "e1Test: ", e1Test.shape
print "e2Test: ", e2Test.shape
print "yTest: ", yTest.shape


f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

print "Embeddings: ",embeddings.shape

########################################### Data Loading finished ######################################################


######################## MODEL with POS and trainable attention and Bidirectional LSTM

class MyLayer(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_shape = (300,300)
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

Input1 = Input(shape=(146,))
distanceModel1 = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(Input1)

Input2 = Input(shape=(146,))
distanceModel2 = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])(Input2)

Input6 = Input(shape=(146,))
POSModel2 =Embedding(len(pos_tags_dict), position_dims, input_length=pos_tags_train.shape[1])(Input6)

Input3 = Input(shape=(146,))
Input4 = Input(shape=(1,))
Input5 = Input(shape=(1,))
wordModel = Input3
wordModel = Embedding(embeddings.shape[0], embeddings.shape[1],
                      input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False)(wordModel)
e1Model = Embedding(embeddings.shape[0], embeddings.shape[1],
                      input_length=1, weights=[embeddings], trainable=False)(Input4)
e2Model = Embedding(embeddings.shape[0], embeddings.shape[1],
                      input_length=1, weights=[embeddings], trainable=False)(Input5)


x_embed = wordModel
e1_embed = e1Model
e1_embed =  Reshape([300,1])(e1_embed)
e2_embed = e2Model
e2_embed =  Reshape([300,1])(e2_embed)

print x_embed.shape
W1x = MyLayer()(x_embed)
print W1x.shape
W2x = MyLayer()(x_embed)
A1 = merge([W1x, e1_embed], output_shape=(146, 1), mode=get_R)
A1 = Lambda(lambda x: x / 300)(A1)
print A1.shape
A2 = merge([W2x, e2_embed], output_shape=(146, 1), mode=get_R)
A2 = Lambda(lambda x: x / 300)(A2)
print A2.shape
alpha1 = Activation("softmax")(A1)
alpha2 = Activation("softmax")(A2)
alpha = average([alpha1, alpha2])
alpha = Flatten()(alpha)
alpha = RepeatVector(300)(alpha)
alpha = Reshape([146, 300])(alpha)
print alpha.shape
att_output = multiply([x_embed, alpha])


c_models = merge([distanceModel1, distanceModel2, POSModel2, att_output], mode='concat', concat_axis=-1)

c_models = Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='tanh',
                        subsample_length=1)(c_models)

c_models = Bidirectional(LSTM(150))(c_models)
c_models = Dropout(0.25)(c_models)
c_models = Dense(n_out, activation='softmax')(c_models)

main_model = Model(inputs=[Input1, Input2, Input6, Input3, Input4, Input5], outputs=c_models)
main_model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
main_model.summary()
SVG(model_to_dot(main_model, show_shapes="True", show_layer_names="False").create(prog='dot', format='svg'))


################################################## Model Complete ####################################################


################################################## Training begins ###################################################

print "Start training"
max_prec, max_rec, max_acc, max_f1 = 0,0,0,0
for epoch in xrange(50):
    print epoch
    main_model.fit([positionTrain1, positionTrain2, pos_tags_train, sentenceTrain, e1Train, e2Train], train_y_cat, batch_size=batch_size, verbose=True,nb_epoch=1)
    pred_vals = main_model.predict([ positionTest1, positionTest2, pos_tags_test, sentenceTest, e1Test, e2Test], verbose=False)
    pred_test = []
    for idx in range(len(pred_vals)):
        pred_test.append(np.argmax(pred_vals[idx]))

    pred_test=np.array(pred_test)
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(yTest)

    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)

    f1Sum = 0
    f1Count = 0
    for targetLabel in xrange(0, max(yTest)+1):
        prec = getPrecision(pred_test, yTest, targetLabel)
        rec = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        #print f1
        f1Sum += f1
        f1Count +=1


    macroF1 = f1Sum / float(f1Count)
    max_f1 = max(max_f1, macroF1)
    print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)

################################################## Training terminates ###################################################

main_model.save("attention.model")