languagelist = ['Ar', 'Es']
emotionlist = ['anger', 'fear', 'joy', 'sadness']
typelist = ['train', 'dev']
for language in languagelist:
    for emotion in emotionlist:
        for type in typelist:
            fname = 'data/2018-EI-reg-'+language+'-'+type+'/2018-EI-reg-'+language+'-'+emotion+'-'+type+'.txt'
            tfname = 'translated0/'+language+'_'+emotion+'_'+type+'.txt'
            foutname = 'translated/'+language+'_'+emotion+'_'+type+'.txt'
            with open(fname) as f, open(tfname) as ft, open(foutname, 'w') as fw:
                lines = f.readlines()
                tlines = ft.readlines()
                fw.write(lines[0])
                for line, tline in zip(lines[1:], tlines):
                    line = line.strip().split('\t')
                    outline = [line[0], tline.strip(), line[2], line[3]]
                    fw.write('\t'.join(outline))
                    fw.write('\n')
        fname = 'data/SemEval2018-Task1-AIT-Test-gold/EI-reg/2018-EI-reg-'+language+'-'+emotion+'-test-gold.txt'
        tfname = 'translated0/' + language + '_' + emotion + '_test.txt'
        foutname = 'translated/' + language + '_' + emotion + '_test.txt'
        with open(fname) as f, open(tfname) as ft, open(foutname, 'w') as fw:
            lines = f.readlines()
            tlines = ft.readlines()
            fw.write(lines[0])
            for line, tline in zip(lines[1:], tlines):
                line = line.strip().split('\t')
                outline = [line[0], tline.strip(), line[2], line[3]]
                fw.write('\t'.join(outline))
                fw.write('\n')

