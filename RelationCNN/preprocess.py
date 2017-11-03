"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""

import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip
import os


## Globals

#Mapping of the labels to integers
semeval_labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}
ddi_labelsMapping = {
    'None(e1,e2)' : 0,
    'effect(e1,e2)' : 1,
    'advise(e1,e2)' : 2,
    'mechanism(e1,e2)' : 3,
    'int(e1,e2)' : 4
}

minDistance = -30
maxDistance = 30




def createMatrices(file, word2Idx, labelsDistribution, labelsMapping, distanceMapping, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix = []
    
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens = sentence.split(" ")
        
        labelsDistribution[label] += 1
      
        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in xrange(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
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
            
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        
        labels.append(labelsMapping[label])
        

    
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
        
        
 
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]



"""
For obtaining word embedding features from the two domains, we do the following:
    (1) take the union of the vocabularies from domain 1 and domain 2
    (2) take the intersection of this with the words in the dataset
    (3) order the resulting filtered vocabulary
    (4) build the embedding matrix for each domain as follows:
        a. iterate over the words in the ordered vocabulary
        b. if the word is embedded in that domain, add its embedding
        c. otherwise, use that domain's UNKNOWN embedding
"""

def getVocabularyUnion(embeddingsPath1, embeddingsPath2=None, words=set()):
    vocab = set()
    paths = [embeddingsPath1]
    if not embeddingsPath2 is None: paths.append(embeddingsPath2)
    for embeddingsPath in paths:
        vocab = vocab.union(readEmbeddingVocabulary(embeddingsPath, words))
    return vocab

def readEmbeddingVocabulary(embeddingsPath, words):
    vocab = set()
    for line in open(embeddingsPath):
        split = line.strip().split(" ")
        word = split[0].lower()

        if word in words:
            vocab.add(word)
    return vocab

def getOrderedVocabulary(embeddingsPath1, embeddingsPath2, words):
    union = getVocabularyUnion(embeddingsPath1, embeddingsPath2, words)
    ordered = tuple(union)

    # base cases for word2Idx
    word2Idx = {
        "PADDING": 0,
        "UNKNOWN": 1
    }

    for word in ordered:
        word2Idx[word] = len(word2Idx)

    return word2Idx

def loadFilteredEmbeddings(embeddingsPath, word2Idx):
    # check how many dimensions are in the embedding file
    with open(embeddingsPath) as f:
        first_line = f.readline()
        split = [s.strip() for s in first_line.split(" ")]
        ndim = len(split) - 1  # account for the word itself

    embeddings = np.zeros([len(word2Idx), ndim])
    # add PADDING (this is unnecessary, since already zeros, but makes me feel better)
    embeddings[word2Idx["PADDING"]] = np.zeros(ndim)
    # add UNKNOWN
    embeddings[word2Idx["UNKNOWN"]] = np.random.uniform(-0.25, 0.25, ndim)

    # read in the embeddings specified in the file
    words_seen = set()
    for line in open(embeddingsPath):
        split = [s.strip() for s in line.split(" ")]
        word = split[0].lower()
        assert len(split) == ndim+1

        if word in word2Idx:
            vector = np.array([float(num) for num in split[1:]])
            embeddings[word2Idx[word]] = vector
            words_seen.add(word)

    # copy UNKNOWN embedding for all unseen words
    for word in word2Idx:
        if not word in words_seen:
            embeddings[word2Idx[word]] = embeddings[word2Idx["UNKNOWN"]]
            
    return embeddings


def preprocess(embeddings1Path, embeddings2Path, files, labelsMapping, pkl_dir):
    outputFilePath = '%s/sem-relations.pkl.gz' % pkl_dir
    embeddings1PklPath = '%s/embeddings1.pkl.gz' % pkl_dir
    if embeddings2Path: embeddings2PklPath = '%s/embeddings2.pkl.gz' % pkl_dir

    words = {}
    maxSentenceLen = [0,0,0]
    labelsDistribution = FreqDist()

    distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
    for dis in xrange(minDistance,maxDistance+1):
        distanceMapping[dis] = len(distanceMapping)


    for fileIdx in xrange(len(files)):
        file = files[fileIdx]
        for line in open(file):
            splits = line.strip().split('\t')
            
            label = splits[0]
            
            
            sentence = splits[3]        
            tokens = sentence.split(" ")
            maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
            for token in tokens:
                words[token.lower()] = True
                

    print "Max Sentence Lengths: ",maxSentenceLen
            
    # :: Read in word embeddings ::



    word2Idx = getOrderedVocabulary(embeddings1Path, embeddings2Path, words)
    embeddings1 = loadFilteredEmbeddings(embeddings1Path, word2Idx)
    if embeddings2Path: embeddings2 = loadFilteredEmbeddings(embeddings2Path, word2Idx)

    print "Embeddings (1) shape: ", embeddings1.shape
    if embeddings2Path: print "Embeddings (2) shape: ", embeddings2.shape
    print "Len words: ", len(words)

    output_pairs = [(embeddings1, embeddings1PklPath)]
    if embeddings2Path: output_pairs.append((embeddings2, embeddings2PklPath))
    for (embeddings, embeddingsPklPath) in output_pairs:
        f = gzip.open(embeddingsPklPath, 'wb')
        pkl.dump(embeddings, f, -1)
        f.close()

    # :: Create token matrix ::
    train_set = createMatrices(files[0], word2Idx, labelsDistribution, labelsMapping, distanceMapping, max(maxSentenceLen))
    validation_set = createMatrices(files[1], word2Idx, labelsDistribution, labelsMapping, distanceMapping, max(maxSentenceLen))
    test_set = createMatrices(files[2], word2Idx, labelsDistribution, labelsMapping, distanceMapping, max(maxSentenceLen))



    f = gzip.open(outputFilePath, 'wb')
    pkl.dump(train_set, f, -1)
    pkl.dump(validation_set, f, -1)
    pkl.dump(test_set, f, -1)
    f.close()



    print "Data stored in pkl folder"

    for label, freq in labelsDistribution.most_common(100):
        print "%s : %f%%" % (label, 100*freq / float(labelsDistribution.N()))


if __name__=='__main__':
    folder = 'files/'
    semeval_files = [folder+'train_reduced.txt', folder+'validation.txt', folder+'test.txt']
    ddi_files = [folder+'train_ddi_reduced.txt', folder+'validation_ddi.txt', folder+'test_ddi.txt']
    
    ## SemEval - Gigaword SGNS only
    #pkl_dir = 'pkl_semeval_gigaword_sgns'
    #embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/gigaword.sgns.txt'
    #embeddings2Path = None
    #files = semeval_files
    #labelsMapping = semeval_labelsMapping

    ## SemEval - Wikipedia SGNS only
    #pkl_dir = 'pkl_semeval_wikipedia_sgns'
    #embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/wikipedia.sgns.txt'
    #embeddings2Path = None
    #files = semeval_files
    #labelsMapping = semeval_labelsMapping

    ## DDI - PubMed only
    #pkl_dir = 'pkl_ddi_pubmed'
    #embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/Pubmed.txt'
    #embeddings2Path = None
    #files = ddi_files
    #labelsMapping = ddi_labelsMapping

    ## DDI - PubMed+Gigaword
    #pkl_dir = 'pkl_ddi_pubmed_gigaword'
    ##embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/Pubmed.txt'
    #embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/pubmed.sgns.txt'
    #embeddings2Path = '/fs/project/PAS1315/projgroup7/embeddings/gigaword.sgns.txt'
    #files = ddi_files
    #labelsMapping = ddi_labelsMapping

    # DDI - Gigaword+Pubmed
    pkl_dir = 'pkl_ddi_gigaword_pubmed'
    embeddings1Path = '/fs/project/PAS1315/projgroup7/embeddings/gigaword.sgns.txt'
    embeddings2Path = '/fs/project/PAS1315/projgroup7/embeddings/pubmed.sgns.txt'
    files = ddi_files
    labelsMapping = ddi_labelsMapping

    if not os.path.isdir(pkl_dir):
        os.mkdir(pkl_dir)

    preprocess(embeddings1Path, embeddings2Path, files, labelsMapping, pkl_dir)
