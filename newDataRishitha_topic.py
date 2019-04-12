import os

emotionlist = ['anger', 'fear', 'joy', 'sadness']
#occupationlist = ['bus', 'journ', 'photo', 'student', 'teacher']
infname = 'newDataRishitha/tweet_API_armynavygame.txt'
with open(infname) as f:
    lines = [line.strip() for line in f]
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
    # print('->', inpath+item, newlinelist)

    for emotion in emotionlist:
        outpath = 'newDataProcessed/Rishitha_topic/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        #for occupation in occupationlist:
        fname = emotion + '.txt'
        outfname = outpath + fname
        with open(outfname, 'w') as fw:
            fw.write(lines[0] + '\n')
            for line in newlinelist:
                if line.strip().split('\t')[-2].strip() == emotion:
                    fw.write(line+'\n')



