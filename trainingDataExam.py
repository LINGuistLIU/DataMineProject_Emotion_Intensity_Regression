import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import os

random.seed(1)

emotionlist = ['joy', 'anger', 'fear', 'sadness']
#emotionlist = ['joy']
datapath = 'data/EI-reg-En-train/'
for emotion in emotionlist:
    print('Processling', emotion, '...')
    fname = datapath+'EI-reg-En-'+emotion+'-train.txt'
    intensityCountdict = {}
    with open(fname) as f:
        allLines = f.readlines()
        title = allLines[0]
        lines = allLines[1:]
        intensitylist = [float(line.strip().split('\t')[-1]) for line in lines]
        for intensity in intensitylist:
            if intensity <= 0.1:
                intensityCountdict[0.1] = intensityCountdict.get(0.1, 0) + 1
            elif intensity > 0.1 and intensity <= 0.2:
                intensityCountdict[0.2] = intensityCountdict.get(0.2, 0) + 1
            elif intensity > 0.2 and intensity <= 0.3:
                intensityCountdict[0.3] = intensityCountdict.get(0.3, 0) + 1
            elif intensity > 0.3 and intensity <= 0.4:
                intensityCountdict[0.4] = intensityCountdict.get(0.4, 0) + 1
            elif intensity > 0.4 and intensity <= 0.5:
                intensityCountdict[0.5] = intensityCountdict.get(0.5, 0) + 1
            elif intensity > 0.5 and intensity <= 0.6:
                intensityCountdict[0.6] = intensityCountdict.get(0.6, 0) + 1
            elif intensity > 0.6 and intensity <= 0.7:
                intensityCountdict[0.7] = intensityCountdict.get(0.7, 0) + 1
            elif intensity > 0.7 and intensity <= 0.8:
                intensityCountdict[0.8] = intensityCountdict.get(0.8, 0) + 1
            elif intensity > 0.8 and intensity <= 0.9:
                intensityCountdict[0.9] = intensityCountdict.get(0.9, 0) + 1
            else:
                intensityCountdict[1.0] = intensityCountdict.get(1.0, 0) + 1
    #print(intensityCountdict)
    #plt.hist(intensitylist, normed=True, bins=10)
    #plt.hist(intensitylist, bins=10)
    #plt.ylabel('Count')
    #plt.savefig('plots/joyTrain.pdf')
    #plt.close()
    x = np.arange(10)
    barnamelist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    heightlist = [intensityCountdict[item] for item in barnamelist]
    plt.bar(x, height=heightlist)
    plt.xticks(x, barnamelist)
    plt.title(emotion)
    plt.savefig('plots/'+emotion+'Train_bar.pdf')
    plt.close()

    maxcount = max(heightlist)
    print(maxcount)
    intensityRange_tweetlist_dict = defaultdict(list)
    for line in lines:
        intensity = float(line.strip().split('\t')[-1])
        if intensity <= 0.1:
            intensityRange_tweetlist_dict[0.1].append(line)
        elif intensity > 0.1 and intensity <= 0.2:
            intensityRange_tweetlist_dict[0.2].append(line)
        elif intensity > 0.2 and intensity <= 0.3:
            intensityRange_tweetlist_dict[0.3].append(line)
        elif intensity > 0.3 and intensity <= 0.4:
            intensityRange_tweetlist_dict[0.4].append(line)
        elif intensity > 0.4 and intensity <= 0.5:
            intensityRange_tweetlist_dict[0.5].append(line)
        elif intensity > 0.5 and intensity <= 0.6:
            intensityRange_tweetlist_dict[0.6].append(line)
        elif intensity > 0.6 and intensity <= 0.7:
            intensityRange_tweetlist_dict[0.7].append(line)
        elif intensity > 0.7 and intensity <= 0.8:
            intensityRange_tweetlist_dict[0.8].append(line)
        elif intensity > 0.8 and intensity <= 0.9:
            intensityRange_tweetlist_dict[0.9].append(line)
        else:
            intensityRange_tweetlist_dict[1.0].append(line)

    newtweetline_dict = defaultdict(list)
    for barname in barnamelist:
        if len(intensityRange_tweetlist_dict[barname]) == maxcount:
            newtweetline_dict[barname] = intensityRange_tweetlist_dict[barname]
        else:
            for i in range(maxcount):
                newtweetline_dict[barname].append(random.choice(intensityRange_tweetlist_dict[barname]))
    #for k, v  in newtweetline_dict.items():
    #    print(k, len(v))
    outlist = []
    for k, v in newtweetline_dict.items():
        outlist += v

    foutpath = 'data/en_train/'
    if not os.path.exists(foutpath):
        os.makedirs(foutpath)
    foutname = foutpath+emotion+'_train.txt'
    random.shuffle(outlist)
    with open(foutname, 'w') as fw:
        fw.write(title)
        for line in outlist:
            fw.write(line)

