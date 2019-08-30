import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil


# if 0, find best mask based on metric file. Else uses mask number.
file = '/roi.txt'
all_folders_in_dir = r'C:\Users\wellerj\Desktop\waveopt_oop\run_8-23\compare_masks_8-26_compareall'

if all_folders_in_dir != '':
    folders = []
    for root, dirs, files in os.walk(all_folders_in_dir):
        for d in dirs:
            if 'zopt' in d:
                continue
            f = os.path.join(root,d)
            if os.path.isfile(f+file):
                folders.append(f)
                print(f)

##logfile = 
##file = open(logfile+'/averages.txt','w+')

mets = {'max_metric':[], 'max_intensity':[], 'spot':[], 'mean':[]}
keys = list(mets.keys())
print(keys)

labels = []
for k, folder in enumerate(folders):
    l = folder[folder.find('--')+2:folder.rfind('.')].replace('coeff_0--','').replace('mode_','m')
    if l.find('base') == -1:
        l = l[:l.find('_mea')]
    else:
        l='zernike'
        znum = k
    if l.find('noma') != -1:
        l='nomask'
    
    labels.append(l)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.find('_vals_checkpoint') != -1:
                print('removed:',file[:file.find('_vals')])
                spath = os.path.join(root,file[:file.find('_vals')]+'_plot')
                f = os.path.join(root,file)
                dat = np.loadtxt(f)
                
##                plt.figure()
##                fig = plt.plot(np.arange(dat.shape[0]),dat)
##                plt.savefig(spath)
##                plt.close()

                for key in keys:
                    if file.find(key) != -1:
                        mets[key].append(np.mean(dat))
print(np.shape(mets))
print(mets)

sort = np.argsort(mets['max_metric'])
labels = np.array(labels)[sort]
print('sorted',sort)
print('labels',labels)

for key in keys:
    dat = np.array(mets[key])
    dat = dat/dat[znum]
    dat = dat[sort]
    plt.figure(figsize=(8,6))
    ax = plt.axes()
    plt.bar(np.arange(dat.shape[0]),dat,width=0.5)
    plt.plot(plt.xlim(),[1,1],lw=1,ls='--',c='k')
    plt.ylim([np.min(dat)-.1,plt.ylim()[1]])
    plt.title(key)

    ax.set_xticks(np.arange(dat.shape[0]))
    ax.set_xticklabels(labels, rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(all_folders_in_dir,str(key)+'_compare_plot'),dpi=300)

plt.show()

            
                    
