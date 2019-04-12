import os


inpath = 'newDataBP/'

for item in os.listdir(inpath):
    if 'JOY' in item:
        outpath = 'newDataProcessed/BP/joy/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if 'ART' in item:
            outfile = outpath+'photo_joy.txt'
            with open(inpath+item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')
        if 'BUSS' in item:
            outfile = outpath+'bus_joy.txt'
            with open(inpath+item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

        if 'JOURNA' in item:
            outfile = outpath+'journ_joy.txt'
            with open(inpath+item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

        if 'STUD' in item:
            outfile = outpath+'student_joy.txt'
            with open(inpath+item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

        if 'TEACH' in item:
            outfile = outpath+'teacher_joy.txt'
            with open(inpath+item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

    if 'SAD' in item:
        outpath = 'newDataProcessed/BP/sadness/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if 'ART' in item:
            outfile = outpath + 'photo_sadness.txt'
            with open(inpath + item, encoding='ISO-8859-1') as f, open(outfile, 'w', encoding='utf-8') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')
        if 'BUSS' in item:
            outfile = outpath + 'bus_sadness.txt'
            with open(inpath + item, encoding='ISO-8859-1') as f, open(outfile, 'w', encoding='utf-8') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                i = 0
                for line in lines:
                    i += 1
                    linesplit = line.strip().split('\t')
                    if len(linesplit) == 4:
                        outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                        fw.write('\t'.join(outlist))
                        fw.write('\n')
                    else:
                        print(outfile, 'LINE',i, linesplit)
                        outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], 'sadness', 'none']
                        fw.write('\t'.join(outlist))
                        fw.write('\n')

        if 'JOUR' in item:
            outfile = outpath + 'journ_sadness.txt'
            with open(inpath + item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

        if 'STUD' in item:
            outfile = outpath + 'student_sadness.txt'
            with open(inpath + item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                for line in lines:
                    linesplit = line.strip().split('\t')
                    outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                    fw.write('\t'.join(outlist))
                    fw.write('\n')

        if 'TEACH' in item:
            outfile = outpath + 'teacher_sadness.txt'
            with open(inpath + item) as f, open(outfile, 'w') as fw:
                fw.write('\t'.join(['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']))
                fw.write('\n')
                lines = f.readlines()[1:]
                i = 0
                for line in lines:
                    i += 1
                    linesplit = line.strip().split('\t')
                    if len(linesplit) == 4:
                        outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], linesplit[2], linesplit[3]]
                        fw.write('\t'.join(outlist))
                        fw.write('\n')
                    else:
                        print(outfile, 'LINE',i, linesplit)
                        outlist = [linesplit[0].strip()[3:], linesplit[1].strip()[6:-1], 'sadness', 'none']
                        fw.write('\t'.join(outlist))
                        fw.write('\n')

