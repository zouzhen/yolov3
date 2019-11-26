import os
import shutil
'''
本脚本实现了对不同文件的脚本
'''
verification_tar = 'N'
verification_ori = 'N'
while verification_tar != 'Y':
    target = input('请输入合并的目标文件夹：')
    verification_tar = input('确认合并的目标文件夹位置为 %s\n 确认输入Y，反之输入N：'%target)
    print('文件夹%s文件数为%d'%(target,len(os.listdir(target))))
while verification_ori != 'Y':
    origin = input('请输入合并的源文件夹：')
    verification_ori = input('确认合并的源文件夹位置为 %s\n 确认输入Y，反之输入N：'%origin)
    print('文件夹%s文件数为%d'%(target,len(os.listdir(origin))))


if verification_tar == 'Y' and verification_ori == 'Y':
    origin_file_list = os.listdir(origin)
    for file in files:
        path = os.path.join(origin,file)
        if os.path.exists(path):
            shutil.move(path,target)

print('合并后%s文件夹内的文件总数为%d'%(target,len(os.listdir(target))))