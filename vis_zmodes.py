import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil


folder = r'C:\Users\wellerj\Desktop\waveopt_oop\run_8-28'

zmodes = []
labels = []

for root, dirs, files in os.walk(folder):
    for d in dirs:
        f = os.path.join(root,d,'optimized_zmodes.txt')
        if os.path.isfile(f):
            zmodes.append(np.loadtxt(f))
            labels.append(d)

zmodes = np.array(zmodes)
print(zmodes.shape)
print(labels)


rows = 4
fig, axs = plt.subplots(rows,zmodes.shape[1]%4+1, figsize=(12,8))

i=0
for r in axs:
    for ax in r:
        if i>=12:
            continue
        ax.hist(zmodes[:,i], align='mid')
        m = np.mean(zmodes[:,i])
        ax.plot([m,m], [0,5], color='r', linewidth=2)
        ax.title.set_text('mode '+str(i+3))
        i+=1

fig.tight_layout()

plt.figure()
plt.scatter(np.arange(zmodes.shape[1]),np.mean(zmodes,axis=0))

print(np.mean(zmodes,axis=0))

plt.show()
