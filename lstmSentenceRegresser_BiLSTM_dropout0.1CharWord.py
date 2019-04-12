# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataLoaderRegresser
import os
import random

#torch.manual_seed(1)
#random.seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#class LSTMClassifier(nn.Module):
    #def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
class LSTMRegresser(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_output, bidirectional=True, dropout=0.1, num_layers=2):
    #def __init__(self, embedding_dim, hidden_dim, vocab_size, n_output):
        #super(LSTMClassifier, self).__init__()
        super(LSTMRegresser, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, dropout=dropout, num_layers=num_layers) #hidden layer
        #self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.predict = nn.Linear(hidden_dim*2, n_output) # output layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell c
        #return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim, device=device)),
        #        autograd.Variable(torch.zeros(1, 1, self.hidden_dim, device=device)))
        return (autograd.Variable(torch.zeros(4, 1, self.hidden_dim, device=device)),
                autograd.Variable(torch.zeros(4, 1, self.hidden_dim, device=device)))
        #return (autograd.Variable(torch.zeros(6, 1, self.hidden_dim, device=device)),
        #        autograd.Variable(torch.zeros(6, 1, self.hidden_dim, device=device)))


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        # https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        # The view function is meant to reshape the tensor. Note that after reshape, the total number of elements need to remain the same.
        # What's the meaning of parameter -1?
        # When you don't know how many rows or columns you want, but are sure of the number of columns or row,
        # you can specify the parameter that you don't know as -1.

        #x = F.relu(x) # activation function after the embedding before going to the LSTM
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #y = self.hidden2label(lstm_out[-1])
        y = self.predict(lstm_out[-1]) #lniear output
        y = torch.sigmoid(y)
        #print('->y', y)
        #log_probs = F.log_softmax(y, dim=1) #https://discuss.pytorch.org/t/implicit-dimension-choice-for-softmax-warning/12314/10
        #log_probs = F.softmax(y, dim=1)
        #print('->log_probs', log_probs)
        #return log_probs
        return y

#def get_accuracy(truth, pred):
#    assert len(truth) == len(pred)
#    right = 0.0
#    for i in range(len(truth)):
#        if truth[i]==pred[i]:
#            right += 1
#    return right/len(truth)


def train_epochChar(model, train_data, loss_function, optimizer, word_to_ix, i, all_losses):
    model.train()
    # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    # model.train() tells your model that you are training the model.
    # So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedure know what is going on and hence can behave accordingly.
    # You can call either model.eval() or model.train(mode=False) to tell that you are testing.

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []

    loss_plot = 0.0
    #for sent, label in train_data:
    random.shuffle(train_data)
    train_data = train_data[:int(0.7*len(train_data))]
    for sentwords, dgr in train_data:
        dgr = dataLoaderRegresser.prepare_degree(dgr)
        #truth_res.append(label_to_ix[label])
        truth_res.append(dgr)
        #detaching it from its history on the last instance
        model.hidden = model.init_hidden()
        sent = dataLoaderRegresser.prepare_sequence_char(sentwords, word_to_ix)
        #label = dataLoader.prepare_label(label, label_to_ix)
        pred = model(sent)
        ######print('->gold-degree %.4f, predicted-degree %.4f %s' % (dgr.item(), pred.item(), sentwords))
        #pred_label = pred.data.max(1)[1].numpy()
        #pred_res.append(pred_label)
        #print('->pred:', pred)
        #print('->dgr:', dgr)
        pred_res.append(pred)
        #model.zero_grad() # set gradients of all model parameters to zero; same as optimizer.zero_grad()
        optimizer.zero_grad()
        # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/2
        # model.zero_grad() vs optimizer.zero_grad()
        # if optimizer = optim.SGD(model.parameters()), model.zero_grad() and optimizer.zero_grad are the same.
        # They are still the same whether the optimizer is SGD, Adam, RMSProp etc.
        #loss = loss_function(pred, label)
        loss = loss_function(pred, dgr)
        #print('->loss', loss.item())
        #avg_loss += loss.data[0]
        avg_loss += loss.item() # https://github.com/pytorch/pytorch/issues/6061
        loss_plot += loss.item()
        count += 1
        printEvery = 50.0
        #if count % 500 == 0: #print out every 500 sentences
        if count % printEvery == 0: #print out every 50 sentences
            #print('epoch: %d iterations: %d loss: %g' % (i, count, loss.data[0]))
            #all_losses.append(loss_plot/printEvery)
            #print('epoch: %d iterations: %d loss: %g' % (i, count, loss.item()))
            print('epoch: %d iterations: %d loss: %g' % (i, count, loss_plot/printEvery))
            loss_plot = 0.0

        loss.backward() # Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
                        # This is why you should call optimizer.zero_grad() after each .step() call.
                        # Note that following the thirst .backward call, a second call is only possible after you have performed another forward pass.
        optimizer.step() # It performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.
    avg_loss /= len(train_data)
    all_losses.append(avg_loss)
    print('epoch: %d done!\ntrain avg_loss: %g' % (i, avg_loss))
    # %g
    # https://stackoverflow.com/questions/30580481/why-does-e-behave-different-than-g-in-format-strings
    return all_losses, model

def evaluateChar(model, data, loss_function, word_to_ix, all_losses_dev, name='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    loss_plot = 0.0
    count = 0.0
    for sentwords, dgr in data:
        count += 1
        dgr = dataLoaderRegresser.prepare_degree(dgr)
        truth_res.append(dgr)
        #detaching it from its history on the last instance
        model.hidden = model.init_hidden()
        sent = dataLoaderRegresser.prepare_sequence_char(sentwords, word_to_ix)
        pred = model(sent)
        ######print('->gold-degree %.4f, predicted-degree %.4f %s' % (dgr.item(), pred.item(), sentwords))
        pred_res.append(pred)
        #model.zero_grad() # Note that we don't need to keep this when evaluating the model
        #loss = loss_function(pred, label)
        loss = loss_function(pred, dgr)
        avg_loss += loss.item()
        loss_plot += loss.item()
        plotEvery = 10.0
        if count % plotEvery == 0:
            #all_losses_dev.append(loss_plot/plotEvery)
            loss_plot = 0.0
    avg_loss /= len(data)
    all_losses_dev.append(avg_loss)
    print(name + ' avg_loss: %g' % avg_loss)
    return avg_loss, pred_res, all_losses_dev

def trainChar(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, EPOCH=100):
    best_dev_avgloss = 1.0
    no_drop = 0
    all_losses = []
    all_losses_dev = []
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        all_losses, model = train_epochChar(model, train_data, loss_Function, optimizer, word_to_ix, i, all_losses)

        print('now best dev loss:', best_dev_avgloss)
        dev_avgloss, dev_pred_res, all_losses_dev = evaluateChar(model, dev_data, loss_Function, word_to_ix, all_losses_dev, 'dev')
        if dev_avgloss < best_dev_avgloss:
            best_dev_avgloss = dev_avgloss
            os.system('rm best_models/best_model_avgloss_*.model')
            print('New Best Dev!!!', best_dev_avgloss)
            torch.save(model.state_dict(), 'best_models/best_model_avgloss_'+str(int(dev_avgloss*10000))+'.model')
            no_drop = 0.0
        else:
            no_drop += 1
            if no_drop >= 20:
                break
    all_losses_test = []
    test_avgloss, test_pred_res, _ = evaluateChar(model, test_data, loss_Function, word_to_ix, all_losses_test, 'test')
    return test_pred_res, all_losses, all_losses_dev, best_dev_avgloss

def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, i, all_losses):
    model.train()
    # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    # model.train() tells your model that you are training the model.
    # So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedure know what is going on and hence can behave accordingly.
    # You can call either model.eval() or model.train(mode=False) to tell that you are testing.

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []

    loss_plot = 0.0
    #for sent, label in train_data:
    random.shuffle(train_data)
    train_data = train_data[:int(0.7*len(train_data))]
    for sentwords, dgr in train_data:
        dgr = dataLoaderRegresser.prepare_degree(dgr)
        #truth_res.append(label_to_ix[label])
        truth_res.append(dgr)
        #detaching it from its history on the last instance
        model.hidden = model.init_hidden()
        sent = dataLoaderRegresser.prepare_sequence(sentwords, word_to_ix)
        #label = dataLoader.prepare_label(label, label_to_ix)
        pred = model(sent)
        ######print('->gold-degree %.4f, predicted-degree %.4f %s' % (dgr.item(), pred.item(), sentwords))
        #pred_label = pred.data.max(1)[1].numpy()
        #pred_res.append(pred_label)
        #print('->pred:', pred)
        #print('->dgr:', dgr)
        pred_res.append(pred)
        #model.zero_grad() # set gradients of all model parameters to zero; same as optimizer.zero_grad()
        optimizer.zero_grad()
        # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/2
        # model.zero_grad() vs optimizer.zero_grad()
        # if optimizer = optim.SGD(model.parameters()), model.zero_grad() and optimizer.zero_grad are the same.
        # They are still the same whether the optimizer is SGD, Adam, RMSProp etc.
        #loss = loss_function(pred, label)
        loss = loss_function(pred, dgr)
        #print('->loss', loss.item())
        #avg_loss += loss.data[0]
        avg_loss += loss.item() # https://github.com/pytorch/pytorch/issues/6061
        loss_plot += loss.item()
        count += 1
        printEvery = 50.0
        #if count % 500 == 0: #print out every 500 sentences
        if count % printEvery == 0: #print out every 50 sentences
            #print('epoch: %d iterations: %d loss: %g' % (i, count, loss.data[0]))
            #all_losses.append(loss_plot/printEvery)
            print('epoch: %d iterations: %d loss: %g' % (i, count, loss_plot/printEvery))
            loss_plot = 0.0
        loss.backward() # Calling .backward() multiple times accumulates the gradient (by addition) for each parameter.
                        # This is why you should call optimizer.zero_grad() after each .step() call.
                        # Note that following the thirst .backward call, a second call is only possible after you have performed another forward pass.
        optimizer.step() # It performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.
    avg_loss /= len(train_data)
    all_losses.append(avg_loss)
    print('epoch: %d done!\ntrain avg_loss: %g' % (i, avg_loss))
    # %g
    # https://stackoverflow.com/questions/30580481/why-does-e-behave-different-than-g-in-format-strings
    return all_losses, model

#def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name='dev'):
def evaluate(model, data, loss_function, word_to_ix, all_losses_dev, name='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    #print(data)
    #for sent, label in data:
    loss_plot = 0.0
    count = 0.0
    for sentwords, dgr in data:
        count += 1
        dgr = dataLoaderRegresser.prepare_degree(dgr)
        #truth_res.append(label_to_ix[label])
        truth_res.append(dgr)
        #detaching it from its history on the last instance
        model.hidden = model.init_hidden()
        sent = dataLoaderRegresser.prepare_sequence(sentwords, word_to_ix)
        #label = dataLoader.prepare_label(label, label_to_ix)
        pred = model(sent)
        #pred_label = pred.data.max(1)[1].numpy()
        #pred_res.append(pred_label)
        ######print('->gold-degree %.4f, predicted-degree %.4f %s' % (dgr.item(), pred.item(), sentwords))
        pred_res.append(pred)
        #model.zero_grad() # Note that we don't need to keep this when evaluating the model
        #loss = loss_function(pred, label)
        loss = loss_function(pred, dgr)
        #avg_loss += loss.data[0]
        avg_loss += loss.item()
        loss_plot += loss.item()
        plotEvery = 10.0
        if count % plotEvery == 0:
            #all_losses_dev.append(loss_plot/plotEvery)
            loss_plot = 0.0
    avg_loss /= len(data)
    all_losses_dev.append(avg_loss)
    #acc = get_accuracy(truth_res, pred_res)
    #print(name + ' avg_loss: %g train acc: %g' % (avg_loss, acc))
    print(name + ' avg_loss: %g' % avg_loss)
    return avg_loss, pred_res, all_losses_dev


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plotLosses(all_losses, all_losses_dev, pltname):
    plt.figure()
    plt.plot(all_losses, label='train')
    plt.plot(all_losses_dev, label='dev')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(pltname)
    plt.close()
    return

#def train(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, label_to_ix, best_char_dev_loss=best_char_dev_loss, EPOCH = 100):
def train(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, best_char_dev_avgloss, EPOCH=100):
    #best_dev_acc = 0.0
    #print('->dev_data:', dev_data)
    best_dev_avgloss = best_char_dev_avgloss
    #no_up = 0
    no_drop = 0
    all_losses = []
    all_losses_dev = []
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        #train_epoch(model, train_data, loss_Function, optimizer, word_to_ix, label_to_ix, i)
        all_losses, model = train_epoch(model, train_data, loss_Function, optimizer, word_to_ix, i, all_losses)
        #print('now best dev acc:', best_dev_acc)
        print('now best dev loss:', best_dev_avgloss)
        #dev_acc = evaluate(model, dev_data, loss_Function, word_to_ix, label_to_ix, 'dev')
        #test_acc = evaluate(model, test_data, loss_Function, word_to_ix, label_to_ix, 'test')
        dev_avgloss, dev_pred_res, all_losses_dev = evaluate(model, dev_data, loss_Function, word_to_ix, all_losses_dev, 'dev')
        #test_avgloss = evaluate(model, test_data, loss_Function, word_to_ix, 'test')
        #if dev_acc > best_dev_acc:
        if dev_avgloss < best_dev_avgloss:
            #best_dev_acc = dev_acc
            best_dev_avgloss = dev_avgloss
            os.system('rm best_models/best_model_avgloss_*.model')
            print('New Best Dev!!!', best_dev_avgloss)
            #torch.save(model.state_dict(), 'best_models/mr_best_model_acc_'+str(int(test_acc*10000)) + '.model')
            torch.save(model.state_dict(), 'best_models/best_model_avgloss_'+str(int(dev_avgloss*10000))+'.model')
            #no_up = 0.0
            no_drop = 0.0
        else:
            #no_up += 1
            no_drop += 1
            #if no_up >= 10:
            if no_drop >= 20:
                #exit()
                break
    all_losses_test = []
    test_avgloss, test_pred_res, _ = evaluate(model, test_data, loss_Function, word_to_ix, all_losses_test, 'test')
    return test_pred_res, all_losses, all_losses_dev

if __name__ == '__main__':

    emotionlist = ['joy', 'anger', 'fear', 'sadness']
    #emotionlist = ['joy', 'anger', 'fear', 'sadness']
    #emotionlist = ['joy']
    seedlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for emotion in emotionlist:
        trainfile = 'data/EI-reg-En-train/EI-reg-En-' + emotion + '-train.txt'
        devfile = 'data/2018-EI-reg-En-dev/2018-EI-reg-En-' + emotion + '-dev.txt'
        # testfile = 'data/2018-EI-reg-En-test/2018-EI-reg-En-'+emotion+'-test.txt'
        testfile = 'data/SemEval2018-Task1-AIT-Test-gold/EI-reg/2018-EI-reg-En-' + emotion + '-test-gold.txt'
        #train_data, dev_data, test_data, word_to_ix = dataLoaderRegresser.loadData(trainfile, devfile, testfile)
        train_data, dev_data, test_data, word_to_ix, char_to_ix = dataLoaderRegresser.loadDataChar(trainfile, devfile, testfile)
        for SEED in seedlist:
            torch.manual_seed(SEED)
            random.seed(SEED)
            print('EMOTION:', emotion, 'SEED:', SEED)

            train_dev_data = train_data + dev_data
            random.shuffle(train_dev_data)
            train_data = train_dev_data[:int(len(train_dev_data)*0.9)]
            dev_data = train_dev_data[int(len(train_dev_data)*0.9):]
            #test_data = test_data[:1106]
            print('-> len(test_data): ', len(test_data))
            print('-> test_data example:', test_data[0])
            print('-> test_data example: ', test_data[-1])

            #EMBEDDING_DIM = 50
            #HIDDEN_DIM = 50
            EMBEDDING_DIM = 300
            HIDDEN_DIM = 300
            #EPOCH = 10
            EPOCH = 100
            #best_dev_acc = 0.0
            #model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
            #                       vocab_size=len(word_to_ix), label_size=len(label_to_ix))
            model = LSTMRegresser(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                                  vocab_size=len(word_to_ix), n_output=1)
            model = model.to(device)
            #loss_Function = nn.NLLLoss()
            loss_Function = nn.MSELoss()
            loss_Function = loss_Function.to(device)
            optimizer = optim.Adam(model.parameters(), lr = 0.0001)
            #optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0.01)
            #optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.9)
            #optimizer = optim.SGD(model.parameters(), lr=0.01)

            #train(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, label_to_ix, EPOCH=100)
            #test_pred_res = train(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, EPOCH=2)

            test_pred_res, all_losses, all_losses_dev, best_char_dev_avgloss = trainChar(model, loss_Function, optimizer, train_data, dev_data, test_data, char_to_ix, EPOCH=EPOCH)

            plotpath = 'plots/char/'
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            plotLosses(all_losses, all_losses_dev, plotpath+'en-' + emotion + str(SEED) + '.pdf')


            predpath = 'predictions/char/100Epoch20EarlyStop/'
            if not os.path.exists(predpath):
                os.makedirs(predpath)
            # predfile = 'predictions/en-'+emotion+str(SEED)+'.txt'
            predfile = predpath+'en-' + emotion + str(SEED) + '.txt'
            dataLoaderRegresser.outputPred(testfile, predfile, test_pred_res)

            #--------------------------------------------------------------------------------------------------------------------------

            test_pred_res, all_losses, all_losses_dev = train(model, loss_Function, optimizer, train_data, dev_data, test_data, word_to_ix, best_char_dev_avgloss=best_char_dev_avgloss, EPOCH=EPOCH)

            plotpath = 'plots/charWord/'
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            plotLosses(all_losses, all_losses_dev, plotpath + 'en-' + emotion + str(SEED) + '.pdf')

            predpath = 'predictions/charWord/100Epoch20EarlyStop/'
            if not os.path.exists(predpath):
                os.makedirs(predpath)
            # predfile = 'predictions/en-'+emotion+str(SEED)+'.txt'
            predfile = predpath + 'en-' + emotion + str(SEED) + '.txt'
            dataLoaderRegresser.outputPred(testfile, predfile, test_pred_res)

