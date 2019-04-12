import os

emotionlist = ['anger', 'fear', 'joy', 'sadness']
occupationlist = ['bus', 'journ', 'photo', 'student', 'teacher']
for emotion in emotionlist:
    outpath = 'newDataProcessed/Rishitha/' + emotion + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for occupation in occupationlist:
        inpath = 'newDataRishitha/'+occupation+'/'
        fname = occupation + '_' + emotion + '.txt'
        infname = inpath + fname
        outfname = outpath + fname
        with open(infname) as f, open(outfname, 'w') as fw:
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



