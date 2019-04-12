import os

emotionlist = ['anger', 'fear', 'joy', 'sadness']
for emotion in emotionlist:
    inpath = 'newData/'+emotion+'/'
    outpath = 'newDataProcessed/'+emotion+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for item in os.listdir(inpath):
        if emotion+'.txt' in item:
            outfname = outpath+item
            with open(inpath+item) as f, open(outfname, 'w') as fw:
                lines = [line.strip() for line in f]
                fw.write(lines[0]+'\n')
                newlinelist = []
                newline = []
                for line in lines[1:]:
                    linesplit = line.strip().split('\t')
                    if len(linesplit) > 0:
                        if linesplit[0].isdigit():
                            if len(newline) > 0:
                                newlinelist.append(' '.join(newline))
                            newline = [line]
                        else:
                            newline.append(line)
                if len(newline) > 0:
                    newlinelist.append(' '.join(newline))
                #print('->', inpath+item, newlinelist)
                for line in newlinelist:
                    fw.write(line+'\n')



