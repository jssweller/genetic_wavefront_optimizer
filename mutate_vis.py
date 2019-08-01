import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, math


h,w = 96,64

mi = 0.05
mf = 0.001
decay = 450


def mutate(gen,h,w,mutate_initial_rate,mutate_final_rate, mutate_decay_factor):
    
    segment_rows = 1024/w
    segment_cols = 768/h

    num_segments = int(segment_rows*segment_cols)
    num_mutations = np.round(num_segments * ((mutate_initial_rate - mutate_final_rate)
                                                * np.exp(-gen / mutate_decay_factor)
                                                + mutate_final_rate)).astype(np.int)
    num_mutations[num_mutations < 1] = 1
    return num_mutations


gens = np.arange(1500)

muts = mutate(gens,h,w,mi,mf,decay)

plt.plot(gens,muts)
plt.show()
