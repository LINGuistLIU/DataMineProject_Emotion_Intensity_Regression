import os

inpath = 'data/'
languagelist = ['Ar', 'Es']
typelist = ['train', 'dev', 'test']
emotionlist = ['anger', 'fear', 'joy', 'sadness']
outpath = 'toTranslate/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

for language in languagelist:
    for tp in typelist:
        for emotion in emotionlist:
            infile = inpath+'2018-EI-reg-'+language+'-'+tp+'/'+'2018-EI-reg-'+language+'-'+emotion+'-'+tp+'.txt'
            outfile = outpath+language+'_'+emotion+'_'+tp+'.txt'
            with open(infile) as f:
                fw = open(outfile, 'w')
                lines = f.readlines()[1:]
                i = 0
                fcount = 0
                for line in lines:
                    i += 1
                    print(line)
                    if i <= 150:
                        fw.write(line.strip().split('\t')[1])
                        fw.write('\n')
                    else:
                        fw.close()
                        i = 0
                        fcount += 1
                        outfile = outpath+language+'_'+emotion+'_'+tp+str(fcount)+'.txt'
                        fw = open(outfile, 'w')
                        fw.write(line.strip().split('\t')[1])
                        fw.write('\n')
            fw.close()

