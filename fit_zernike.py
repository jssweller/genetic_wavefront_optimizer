import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil


def get_polybest(x,y,best_func=np.argmin):
    pfit = np.polyfit(x,y,2)
    p = np.poly1d(pfit)
    frange = np.arange(min(x),max(x),1)
    zbest = best_func(p(frange))
    return int(frange[zbest])

def get_best_zmodes(folder, metric = 'max'):
    metrics = {'spot':'spot_metric_vals_checkpoint.txt', 'max':'max_metric_vals_checkpoint.txt'}

    # get metric datasets
    s = []
    labels = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            f = os.path.join(root,d,metrics[metric])
            if os.path.isfile(f):
                s.append(np.loadtxt(f)[1:])
                labels.append(d)
    s = np.array(s)


    # define parameters for zernike algorithm that produced data
    cnum = 10
    crange = np.arange(-80,80,cnum)
    coarse = crange.shape[0]
    fine = np.arange(-cnum,cnum,2).shape[0]+1

    zmodes = np.arange(3,15)
    zmeans = []
    
    for zmode in zmodes:
        start = (zmode-3)*(fine+coarse)

        # fit data to quadratic and get best zmode
        zbest = []
        for zrun in range(s.shape[0]):
            splot = s[zrun][start:start+coarse]

            if metric == 'max':
                bz = get_polybest(crange,splot,np.argmax)
            else:
                bz = get_polybest(crange,splot, np.argmin)
            zbest.append(bz)

        # remove outliers
        zbest = np.asarray(zbest)
        zstd = 2*np.std(zbest)
        zmean = np.mean(zbest)
        if zstd>0:
            zbest = zbest[zbest > zmean-zstd]
            zbest = zbest[zbest < zmean+zstd]
        zmean = np.mean(zbest)

        # add to list
        zmeans.append(int(zmean))

    return np.array(zmeans)


folder = r'C:\Users\wellerj\Desktop\waveopt_oop\run_8-28'
met = 'max'


maxbest = get_best_zmodes(folder,'max')
print('max\n',maxbest)

spotbest = get_best_zmodes(folder,'spot')
print('\nspot\n',spotbest)

print('\ndiff\n',maxbest-spotbest)

spotmaxbest = ((spotbest+maxbest)/2).astype(np.int)
print('\nspotmax\n',spotmaxbest)


# save best zmodes to file
np.savetxt(os.path.join(folder,'maxbest_zmodes.txt'),maxbest, fmt='%d')
np.savetxt(os.path.join(folder,'spotbest_zmodes.txt'),spotbest, fmt='%d')
np.savetxt(os.path.join(folder,'spotmaxbest_zmodes.txt'),spotmaxbest, fmt='%d')


    

##rows = 4
##fig, axs = plt.subplots(rows,s.shape[1]%4+1, figsize=(12,8))
##
##i=0
##for r in axs:
##    for ax in r:
##        if i>=12:
##            continue
##        ax.hist(s[:,i], align='mid')
##        m = np.mean(s[:,i])
##        ax.plot([m,m], [0,5], color='r', linewidth=2)
##        ax.title.set_text('mode '+str(i+3))
##        i+=1
##
##fig.tight_layout()
##
##plt.figure()
##plt.scatter(np.arange(s.shape[1]),np.mean(s,axis=0))
##
##print(np.mean(s,axis=0))
##
##plt.show()
