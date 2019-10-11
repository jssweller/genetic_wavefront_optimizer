import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil


folders = [r'C:\Users\wellerj\Desktop\waveopt_oop\run_8-21_refbeam40\mode_5_coeff_0\96_64_measure_True']

# if 0, find best mask based on metric file. Else uses mask number.
file = '/max_intensity_vals_checkpoint.txt'
masknum = -1


all_folders_in_dir = r'C:\Users\wellerj\Desktop\waveopt_oop\run_8-23'


if all_folders_in_dir != '':
    folders = []
    loglist = open(os.path.join(all_folders_in_dir,'compare_list.txt'),'w+')
    
    for root, dirs, files in os.walk(all_folders_in_dir):
        for d in dirs:
            if 'zopt' in d:
                continue
            f = os.path.join(root,d)
            if os.path.isfile(f+file):
                folders.append(f)
                loglist.write(f+'\n')
                print(f)
    loglist.close()
            

folders = []

for folder in folders:
    if not os.path.isfile(folder+file) or  not os.path.isfile(folder+'/masks.txt'):
        print(folder,'One of the files not found.')
        continue
    
    segs = folder.split('\\')[-1]
    segs = segs.split('_')[:2]
    segs = np.asarray(segs).astype(np.int)

    masks = np.loadtxt(folder+'/masks.txt')
    if os.path.isfile(folder+'/base_mask.txt'):
        basemask = np.loadtxt(folder+'/base_mask.txt')
    else:
        basemask = 0

    if masknum == 0:
        dat = np.loadtxt(folder+file)
        masknum = np.argmax(dat[int(dat.shape[0]/4):])
        print('Mask number', masknum)

    bestmask = masks[masknum]
    bestmask = bestmask.reshape(int(768/segs[0]),int(1024/segs[1]))

    segtemp = np.ones(segs)
    bestmask = np.kron(bestmask,segtemp)
    bestmask += basemask
    bestmask = bestmask.astype(np.uint8)
    print(bestmask.shape)

    np.savetxt(folder+'/bestmask_max.txt',bestmask.reshape((1,-1)),fmt='%d')
