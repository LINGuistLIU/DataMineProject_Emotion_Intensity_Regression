# -*- coding: utf-8 -*-

#import sys
import torch
import torch.autograd as autograd
import random
#import torch.utils.data as Data
import numpy
import io
import numpy as np

#----------------------------------------------------------------#
# https://github.com/cbaziotis/ekphrasis
# Remember to cite their paper mentioned on github.

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    #terms that will be normalized
    normalized = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
    #terms that will be annotated
    annotate = {'hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'},
    fix_html=True, #fix HTML tokens

    #corpus from which the word statistics are going to be used for word segmentation
    segmenter = 'twitter',

    #corpus from which the word statistics are going to be used for spell correction
    corrector = 'twitter',

    unpack_hashtags=True, #perform word segmentation on hashtags
    unpack_contractions=True, #unpack contractions (can't -> can not)
    spell_correction_elong=False, #spell correction for elongated words

    #select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer = SocialTokenizer(lowercase=True).tokenize,
    #tokenizer = SocialTokenizer(lowercase=False).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text with other expressions.
    # You can pass more than one dictionaries.
    dicts=[emoticons]
)
#----------------------------------------------------------------#



#SEED = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input: a sequence of tokens, and a token-to-index dictionary
# output: a LongTensor variable to encode the squence of idxs

bos = '<s>'
eos = '</s>'
#eos = '<eos>'
unk = '<unk>'

#https://github.com/facebookresearch/MUSE/blob/master/demo.ipynb
def load_vec(emb_path, nmax=50000):
    print('Loading pre-trained word embeddings ...')
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            #if len(word2id) == nmax:
            #    break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    print('Pre-trained word embedding loaded successfully!')
    return embeddings, id2word, word2id

#emb_path = 'pretrainedWE/cc.en.300.vec'
#emb_path = 'pretrainedWE/ntua_twitter_300.txt'
emb_path = 'pretrainedWE/wiki.multi.en.vec'
embeddingsEn, id2wordEn, word2idEn = load_vec(emb_path)

#def load_vectors(fname):
#    print('Loading pre-trained word embeddings ...')
#    pretrainedVec = {}
#    with open(fname) as f:
#        lines = f.readlines()
#        n, d = [int(item) for item in lines[0].strip().split()]
#        for line in lines[1:]:
#            tokens = line.strip().split(' ')
#            pretrainedVec[tokens[0]] = [float(item) for item in tokens[1:]]
#    print('Pre-trained word embedding loaded successfully!')
#    return pretrainedVec

#pretrainedVec = load_vectors('pretrainedWE/cc.en.300.vec')

def prepare_sequence_pretrainedVec(seq, pretrainedEmb=embeddingsEn, word2id=word2idEn):
    #var = autograd.Variable(torch.LongTensor([pretrainedEmb[word2id[w]] for w in seq.split(' ') if w in word2id]))
    #return var.to(device)
    #https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array
    var = torch.from_numpy(np.vstack([pretrainedEmb[word2id[w]] for w in seq.split(' ') if w in word2id])).float()
    return var.to(device)

def prepare_sequence(seq, to_ix):
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var.to(device)

def prepare_sequence_char(seq, to_ix):
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq]))
    return var.to(device)

#def prepare_label(label, label_to_ix):
#    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
#    return var.to(device)
def prepare_degree(dgr):
    # https://discuss.pytorch.org/t/how-to-convert-float64-to-torch-autograd-variable/7227/2
    A = numpy.array([dgr])
    B = autograd.Variable(torch.from_numpy(A)).float() #https://github.com/pytorch/pytorch/issues/2138
    B = B.unsqueeze(0) #https://github.com/lidq92/CNNIQA/issues/1
    return B.to(device)

def build_token_to_ix(sentences):
    token_to_ix = {bos:0, eos:1, unk:2}
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    #token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_character_to_ix(sentences):
    char_to_ix = {bos:0, eos:1, unk:2}
    for sent in sentences:
        #for token in sent.split(' '):
        for char in sent:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)
    return char_to_ix

def oovSetting(sent, droprate=0.1):
    sent = sent.strip().split(' ')
    newsent = []
    for item in sent:
        if random.random() < droprate:
            newsent.append(unk)
        else:
            newsent.append(item)
    return ' '.join(newsent)

def loadData(trainfile, devfile, testfile):
    train_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(trainfile)][1:]
    dev_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(devfile)][1:]
    test_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile)][1:]

    train_data = [('<s> '+' '.join(text_processor.pre_process_doc(sent.strip()))+' </s>', float(dgr)) for sent, dgr in train_data]
    dev_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in dev_data]
    test_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> '+sent.strip()+' <eos>', float(dgr)) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in test_data]

    print('train: %d, dev: %d, test: %d' % (len(train_data), len(dev_data), len(test_data)))

    word_to_ix = build_token_to_ix(([s for s, _ in train_data + dev_data + test_data]))
    print('vocab size: %d' % len(word_to_ix))
    print('loading data done!')
    return train_data, dev_data, test_data, word_to_ix

def loadDataSTandNew(trainfile, devfile, testfile, newfilelist):
    train_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(trainfile)][1:]
    dev_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(devfile)][1:]
    test_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile)][1:]

    train_data = [('<s> '+' '.join(text_processor.pre_process_doc(sent.strip()))+' </s>', float(dgr)) for sent, dgr in train_data]
    dev_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in dev_data]
    test_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> '+sent.strip()+' <eos>', float(dgr)) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in test_data]

    new_data_list = []
    for newfile in newfilelist:
        with open(newfile) as fnew:
            new_data = [(line.strip().split('\t')[1], 0.0) for line in fnew][1:]
            new_data = [('<s> '+' '.join(text_processor.pre_process_doc(sent.strip()))+' </s>', float(dgr)) for sent, dgr in new_data]
            new_data_list.append(new_data)

    print('train: %d, dev: %d, test: %d' % (len(train_data), len(dev_data), len(test_data)))

    word_to_ix = build_token_to_ix(([s for s, _ in train_data + dev_data + test_data]))
    print('vocab size: %d' % len(word_to_ix))
    print('loading data done!')
    return train_data, dev_data, test_data, new_data_list, word_to_ix

def loadDataChar(trainfile, devfile, testfile):
    train_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(trainfile)][1:]
    dev_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(devfile)][1:]
    test_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile)][1:]

    train_data = [('<s> '+' '.join(text_processor.pre_process_doc(sent.strip()))+' </s>', float(dgr)) for sent, dgr in train_data]
    dev_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in dev_data]
    test_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> '+sent.strip()+' <eos>', float(dgr)) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in test_data]

    print('train: %d, dev: %d, test: %d' % (len(train_data), len(dev_data), len(test_data)))

    word_to_ix = build_token_to_ix([s for s, _ in train_data + dev_data + test_data])
    char_to_ix = build_character_to_ix([s for s, _ in train_data+dev_data+test_data])
    print('vocab size: %d' % len(word_to_ix))
    print('char size: %d' % len(char_to_ix))
    print('loading data done!')
    return train_data, dev_data, test_data, word_to_ix, char_to_ix

def loadDataArEs(trainfile, devfile):
    train_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(trainfile)][1:]
    dev_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(devfile)][1:]

    train_data = [('<s> '+sent.strip()+' </s>', float(dgr)) for sent, dgr in train_data]
    dev_data = [('<s> ' + sent.strip() + ' </s>', float(dgr)) for sent, dgr in dev_data]

    print('train: %d, dev: %d' % (len(train_data), len(dev_data)))

    word_to_ix = build_token_to_ix(([s for s, _ in train_data + dev_data]))
    char_to_ix = build_character_to_ix([s for s, _ in train_data + dev_data])
    print('vocab size: %d' % len(word_to_ix))
    print('loading data done!')
    return train_data, dev_data, word_to_ix, char_to_ix


def loadNewData(trainfile, devfile, testfile1, testfile2, testfile3, testfile4, testfile5):
    train_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(trainfile)][1:]
    dev_data = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(devfile)][1:]
    test_data1 = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile1)][1:]
    test_data2 = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile2)][1:]
    test_data3 = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile3)][1:]
    test_data4 = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile4)][1:]
    test_data5 = [(line.strip().split('\t')[1], line.strip().split('\t')[-1]) for line in open(testfile5)][1:]

    #train_data = [('<eos> '+' '.join(text_processor.pre_process_doc(sent.strip()))+' <eos>', float(dgr)) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' <eos>', float(dgr)) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' <eos>', float(dgr)) for sent, dgr in test_data]

    #train_data = [('<eos> '+sent.strip()+' <eos>', float(dgr)) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr)) for sent, dgr in dev_data]
    #test_data1 = [('<eos> ' + sent.strip() + ' <eos>', 0.0) for sent, dgr in test_data1]
    #test_data2 = [('<eos> ' + sent.strip() + ' <eos>', 0.0) for sent, dgr in test_data2]
    #test_data3 = [('<eos> ' + sent.strip() + ' <eos>', 0.0) for sent, dgr in test_data3]
    #test_data4 = [('<eos> ' + sent.strip() + ' <eos>', 0.0) for sent, dgr in test_data4]
    #test_data5 = [('<eos> ' + sent.strip() + ' <eos>', 0.0) for sent, dgr in test_data5]

    train_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in train_data]
    dev_data = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', float(dgr)) for sent, dgr in dev_data]
    test_data1 = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', 0.0) for sent, dgr in test_data1]
    test_data2 = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', 0.0) for sent, dgr in test_data2]
    test_data3 = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', 0.0) for sent, dgr in test_data3]
    test_data4 = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', 0.0) for sent, dgr in test_data4]
    test_data5 = [('<s> ' + ' '.join(text_processor.pre_process_doc(sent.strip())) + ' </s>', 0.0) for sent, dgr in test_data5]

    #train_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in train_data]
    #dev_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in dev_data]
    #test_data = [('<eos> ' + sent.strip() + ' <eos>', float(dgr) * 100) for sent, dgr in test_data]

    print('train: %d, dev: %d, new1: %d, new2: %d, new3: %d, new4: %d, new5: %d' % (len(train_data), len(dev_data), len(test_data1), len(test_data2), len(test_data3), len(test_data4), len(test_data5)))

    word_to_ix = build_token_to_ix(([s for s, _ in train_data + dev_data + test_data1 + test_data2 + test_data3 + test_data4 + test_data5]))
    print('vocab size: %d' % len(word_to_ix))
    print('loading data done!')
    return train_data, dev_data, test_data1, test_data2, test_data3, test_data4, test_data5, word_to_ix


def outputPred(testfile, predfile, pred_res):
    with open(testfile) as testf, open(predfile, 'w') as predf:
        testdata = [line.strip().split('\t') for line in testf]
        predf.write('\t'.join(testdata[0]))
        predf.write('\n')
        for gold, pred in zip(testdata[1:], pred_res):
            if float(gold[-1]) != 0:
                predf.write('\t'.join(gold[:-1]+[str(round(pred.item(), 3))]))
                predf.write('\n')
    return


def outputNewPred(testfile, predfile, pred_res):
    with open(testfile) as testf, open(predfile, 'w') as predf:
        testdata = [line.strip().split('\t') for line in testf]
        predf.write('\t'.join(testdata[0]))
        predf.write('\n')
        for gold, pred in zip(testdata[1:], pred_res):
            #if float(gold[-1]) != 0:
            predf.write('\t'.join(gold[:-1]+[str(round(pred.item(), 3))]))
            predf.write('\n')
    return


