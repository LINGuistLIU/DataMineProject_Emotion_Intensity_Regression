import os

def file2list(fn):
    datalist = []
    with open(fn) as f:
        lines = [line.strip().split('\t') for line in f]
        datalist.append(lines[0])
        for line in lines[1:]:
            datalist.append((line[:-1], float(line[-1])))
    return datalist


#emotionlist = ['anger', 'fear', 'joy', 'sadness']
#emotionlist = ['joy', 'anger', 'fear']
#emotionlist = ['sadness']
#emotionlist = ['joy']
#emotionlist = ['anger']
emotionlist = ['fear']
#seedlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#inpath = 'predictions/0.7train3layers0.1dropout/'
#outpath = 'predictions/avg10_biLSTM3layers0.1dropout/'
#inpath = 'predictions/0.7train0.1dropout3layer_textNormalization/'
#outpath = 'predictions/avg10_biLSTM3layers0.1dropout_textNormalization/'

#inpath = 'predictions/charWord/100Epoch20EarlyStop/'
#outpath = 'predictions/charWord/100Epoch20EarlyStop/avg/'
#inpath = 'predictions/pretrainedWE/100Epoch20EarlyStop/best/'
#outpath = 'predictions/pretrainedWE/100Epoch20EarlyStop/avgBest/'
#inpath = 'predictions/dropout0.1layer2WE/100Epoch20EarlyStop/best/'
#outpath = 'predictions/dropout0.1layer2WE/100Epoch20EarlyStop/avgBest/'
#inpath = 'predictions/pretrainedWE_NTUA/NormalizedTrainData/100Epoch20EarlyStop/best/'
inpath = 'predictions/newData/pretrainedWE_NTUA/100Epoch20EarlyStop/best/'
#outpath = 'predictions/pretrainedWE_NTUA/NormalizedTrainData/100Epoch20EarlyStop/avgBest/'
outpath = 'predictions/newData/newDataPredictedLatest/testData/avgBest/'

if not os.path.exists(outpath):
    os.makedirs(outpath)

#inpath = 'predictions/BiLSTMdropout0.1layer2/0.7trainData_sigmoid_20earlyStop/'
#outpath = 'predictions/avg10_biLSTM2layer0.7trainDataSigmoid20earlyStop/'
#outpath = 'predictions/avgGoodModels_biLSTM2layer0.7trainDataSigmoid20earlyStop/'

"""
emotion = 'anger'
outfile = outpath+'en-'+emotion+'-10avg.txt'
ps1 = file2list(inpath + 'en-' + emotion + '1.txt')
ps2 = file2list(inpath + 'en-' + emotion + '2.txt')
ps3 = file2list(inpath + 'en-' + emotion + '3.txt')
ps4 = file2list(inpath + 'en-' + emotion + '4.txt')
#ps5 = file2list(inpath + 'en-' + emotion + '5.txt')
ps6 = file2list(inpath + 'en-' + emotion + '6.txt')
#ps7 = file2list(inpath + 'en-' + emotion + '7.txt')
ps8 = file2list(inpath + 'en-' + emotion + '8.txt')
ps9 = file2list(inpath + 'en-' + emotion + '9.txt')
ps10 = file2list(inpath + 'en-' + emotion + '10.txt')
with open(outfile, 'w') as fw:
    fw.write('\t'.join(ps1[0]))
    fw.write('\n')
    for p1, p2, p3, p4, p6, p8, p9, p10 in zip(ps1[1:], ps2[1:], ps3[1:], ps4[1:], ps6[1:], ps8[1:], ps9[1:], ps10[1:]):
        out = p1[0]
        dgrlist = [p1[1], p2[1], p3[1], p4[1], p6[1], p8[1], p9[1], p10[1]]
        dgr = round(sum(dgrlist)/len(dgrlist), 3)
        print('->', dgr, dgrlist)
        out.append(str(dgr))
        fw.write('\t'.join(out))
        fw.write('\n')

##############################
emotion = 'fear'
outfile = outpath+'en-'+emotion+'-10avg.txt'
ps1 = file2list(inpath + 'en-' + emotion + '1.txt')
ps2 = file2list(inpath + 'en-' + emotion + '2.txt')
ps3 = file2list(inpath + 'en-' + emotion + '3.txt')
#ps4 = file2list(inpath + 'en-' + emotion + '4.txt')
ps5 = file2list(inpath + 'en-' + emotion + '5.txt')
ps6 = file2list(inpath + 'en-' + emotion + '6.txt')
ps7 = file2list(inpath + 'en-' + emotion + '7.txt')
ps8 = file2list(inpath + 'en-' + emotion + '8.txt')
ps9 = file2list(inpath + 'en-' + emotion + '9.txt')
ps10 = file2list(inpath + 'en-' + emotion + '10.txt')
with open(outfile, 'w') as fw:
    fw.write('\t'.join(ps1[0]))
    fw.write('\n')
    for p1, p2, p3, p5, p6, p7, p8, p9, p10 in zip(ps1[1:], ps2[1:], ps3[1:], ps5[1:], ps6[1:], ps7[1:], ps8[1:], ps9[1:], ps10[1:]):
        out = p1[0]
        dgrlist = [p1[1], p2[1], p3[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1]]
        dgr = round(sum(dgrlist)/len(dgrlist), 3)
        print('->', dgr, dgrlist)
        out.append(str(dgr))
        fw.write('\t'.join(out))
        fw.write('\n')

emotion = 'joy'
outfile = outpath+'en-'+emotion+'-10avg.txt'
#ps1 = file2list(inpath + 'en-' + emotion + '1.txt')
ps2 = file2list(inpath + 'en-' + emotion + '2.txt')
ps3 = file2list(inpath + 'en-' + emotion + '3.txt')
ps4 = file2list(inpath + 'en-' + emotion + '4.txt')
ps5 = file2list(inpath + 'en-' + emotion + '5.txt')
#ps6 = file2list(inpath + 'en-' + emotion + '6.txt')
ps7 = file2list(inpath + 'en-' + emotion + '7.txt')
#ps8 = file2list(inpath + 'en-' + emotion + '8.txt')
#ps9 = file2list(inpath + 'en-' + emotion + '9.txt')
ps10 = file2list(inpath + 'en-' + emotion + '10.txt')
with open(outfile, 'w') as fw:
    fw.write('\t'.join(ps1[0]))
    fw.write('\n')
    for p2, p3, p4, p5, p7, p10 in zip(ps2[1:], ps3[1:], ps4[1:], ps5[1:], ps7[1:], ps10[1:]):
        out = p2[0]
        dgrlist = [p2[1], p3[1], p4[1], p5[1], p7[1], p10[1]]
        dgr = round(sum(dgrlist)/len(dgrlist), 3)
        print('->', dgr, dgrlist)
        out.append(str(dgr))
        fw.write('\t'.join(out))
        fw.write('\n')

emotion = 'sadness'
outfile = outpath+'en-'+emotion+'-10avg.txt'
#ps1 = file2list(inpath + 'en-' + emotion + '1.txt')
ps2 = file2list(inpath + 'en-' + emotion + '2.txt')
#ps3 = file2list(inpath + 'en-' + emotion + '3.txt')
ps4 = file2list(inpath + 'en-' + emotion + '4.txt')
ps5 = file2list(inpath + 'en-' + emotion + '5.txt')
#ps6 = file2list(inpath + 'en-' + emotion + '6.txt')
ps7 = file2list(inpath + 'en-' + emotion + '7.txt')
#ps8 = file2list(inpath + 'en-' + emotion + '8.txt')
#ps9 = file2list(inpath + 'en-' + emotion + '9.txt')
ps10 = file2list(inpath + 'en-' + emotion + '10.txt')
with open(outfile, 'w') as fw:
    fw.write('\t'.join(ps1[0]))
    fw.write('\n')
    for p2, p4, p5, p7, p10 in zip(ps2[1:], ps4[1:], ps5[1:], ps7[1:], ps10[1:]):
        out = p2[0]
        dgrlist = [p2[1], p4[1], p5[1], p7[1], p10[1]]
        dgr = round(sum(dgrlist)/len(dgrlist), 3)
        print('->', dgr, dgrlist)
        out.append(str(dgr))
        fw.write('\t'.join(out))
        fw.write('\n')
"""

for emotion in emotionlist:
    outfile = outpath+'en-'+emotion+'-10avg.txt'
    ps1 = file2list(inpath + 'en-' + emotion + '1.txt')
    ps2 = file2list(inpath + 'en-' + emotion + '2.txt')
    ps3 = file2list(inpath + 'en-' + emotion + '3.txt')
    ps4 = file2list(inpath + 'en-' + emotion + '4.txt')
    ps5 = file2list(inpath + 'en-' + emotion + '5.txt')
    ps6 = file2list(inpath + 'en-' + emotion + '6.txt')
    ps7 = file2list(inpath + 'en-' + emotion + '7.txt')
    ps8 = file2list(inpath + 'en-' + emotion + '8.txt')
    ps9 = file2list(inpath + 'en-' + emotion + '9.txt')
    ps10 = file2list(inpath + 'en-' + emotion + '10.txt')
    with open(outfile, 'w') as fw:
        fw.write('\t'.join(ps1[0]))
        fw.write('\n')
        for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 in zip(ps1[1:], ps2[1:], ps3[1:], ps4[1:], ps5[1:], ps6[1:], ps7[1:], ps8[1:], ps9[1:], ps10[1:]):
            out = p1[0]
            dgrlist = [p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1]]
            dgr = round(sum(dgrlist)/len(dgrlist), 3)
            print('->', dgr, dgrlist)
            out.append(str(dgr))
            fw.write('\t'.join(out))
            fw.write('\n')



