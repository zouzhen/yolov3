import os
import shutil
import random
filepath = '/media/lzc274500/Elements SE/西安仪表盘演示/西安仪表'
imagepath = '/media/lzc274500/Elements SE/西安仪表盘演示/JPEGImages'
xmlpath = '/media/lzc274500/Elements SE/西安仪表盘演示/Annotations'
filelist = os.listdir(filepath)
totalfilelist = []
letter = ['A','B','C','D','E','F','G','H','J','K','L','P','R','S','U','V']
num = ['1','2','3','4','5','6','7','8','9','0']
concate = letter + num
numcount = 0
for filename in filelist:
    name = ''.join(random.sample(concate,5))
    subfile = os.path.join(filepath,filename)
    if os.path.isdir(subfile):
        subfilelist = os.listdir(subfile)
        subfilexml = [i for i in subfilelist if i.endswith('.xml')]
    if len(subfilexml) != 0:
        for i in subfilexml:
            shutil.copy2(os.path.join(subfile,i),os.path.join(xmlpath,name + '%08d' % numcount + '.xml'))
            try:
                shutil.copy2(os.path.join(subfile, i.split(".")[0] + '.jpg'),os.path.join(imagepath,name + '%08d' % numcount + '.jpg'))
            except Exception:
                pass
            numcount = numcount + 1