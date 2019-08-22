import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil
import datetime as dt

import Optimizer, Interface, Population, textwrap


def main(args):
    start_num = 'run_mask_comparison1'
    run_description = 'Same as 8-19_baseline_refbeam run but with mirror \
in place of SLM.'
    
##    folder = '../Compare_Masks/run_'+str(start_num)
##    os.makedirs(folder,exist_ok=True)
##    shutil.copy('./take_zernike_data.py',folder+'/mainscript.py')
##    shutil.copystat('./take_zernike_data.py',folder+'/mainscript.py')
##
##    file = open(folder+'/log.txt','w+')
##    print('Run Description: ',run_description)
##    file.write('Description: '+run_description+'\n\n')
##    file.close()
    
##    interface = Interface.Interface(args)

##    args.mutate_initial_rate = 0.01
##    args.mutate_final_rate = 0.001
##    args.mutate_decay_factor = 450
##    
##    args.num_initial_metrics = 500
##    args.num_masks = 20
##    args.num_childs = 15
##    args.fitness_func = 'max'
##    args0 = copy.copy(args)
##    
##    args = copy.copy(args0)
##    zopt = Optimizer.Optimizer(args,interface)
##    baseline_frames = 100
##    num_to_average = 500
##    run_minutes = 18*60
    shape = [768,1024]
    gfile = ''
    zfile = ''
    
    genetic_mask = np.loadtxt(gfile, dtype=np.uint8).reshape(shape)
    zernike_mask = np.loadtxt(zfile, dtype=np.uint8).reshape(shape)
    masks = [genetic_mask,zernike_mask]
    
    labels = ['genetic','zernike']
    rois = [[],[]]
    times = [[],[]]
    
    ti,tf = 22,4

    t0 = dt.datetime.now()
    start_time = dt.datetime.combine(t0.date(),dt.time(ti))
    end_time = dt.datetime.combine((t0+dt.timedelta(1)).date(),dt.time(tf))

    print(start_time,end_time,dt.time())

    while start_time > dt.datetime.now():
        print('WAITING... Time left before start:', start_time - dt.datetime.now())
        time.sleep(5)

    
    numframes = 500
    while end_time > dt.datetime.now():
        rnd = np.random.randint(0,2)
        zopt.base_mask = masks[rnd]
        times[rnd].append(dt.datetime.now())
        rois[rnd].append(zopt.get_baseline_intensity(numframes))

    times = np.asarray(times)
    rois = np.asarray(rois)

    for i,label in enumerate(labels):
        fdir = folder+'/'+label
        zopt.save_rois_metrics(rois[i,:], save_path=fdir)
        np.savetxt(fdir+'/baseline_times.txt', times[i,:], np.asarray(times,dtype='datetime64[s]'), fmt='%s')
        

        
        
        

    
##    zopt.get_baseline_intensity(baseline_frames)
##    zopt.get_baseline_maxmean(run_minutes, num_to_average)
##    file = open(folder+'/log.txt','a')
##    file.write('baseline time: '+str(zopt.get_time()))
##    file.write('\nbaseline frames: '+str(baseline_frames))
##    file.write('\nbaseline num_to_average: '+str(num_to_average))
##    file.close()


            
if __name__ == '__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--help':
        print(__doc__)
    if len(sys.argv)==2 and sys.argv[1]=='--info':
        print(__doc__)

    # Parse Command Line Arguments   
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--pipe_in_handle',
        type=str,
        default='\\\\.\\pipe\\LABVIEW_OUT',
        help='Input Pipe handle. DEFAULT="\\\\.\\pipe\\LABVIEW_OUT"'
    )
    parser.add_argument(
        '--pipe_out_handle',
        type=str,
        default='\\\\.\\pipe\\LABVIEW_IN',
        help='Output Pipe handle. DEFAULT="\\\\.\\pipe\\LABVIEW_IN"'
    )        
    parser.add_argument(
        '--bytes_buffer_size',
        type=int,
        default=4,
        help='Number of bytes describing the input/output buffer size. DEFAULT=4'
    )  
    parser.add_argument(
        '--plot',
        type=bool,
        default=True,
        help='Turn on/off visualization of optimization. DEFAULT=False'
    )
    parser.add_argument(
        '--add_uniform_childs',
        type=bool,
        default=False,
        help='Turn on/off visualization of optimization. DEFAULT=False'
    )
    parser.add_argument(
        '--measure_all',
        type=bool,
        default=True,
        help='Toggle whether or not to measure all masks each generation. DEFAULT=True'
    )
    parser.add_argument(
        '--slm_width',
        type=int,
        default=1024,
        help='Pixel width of SLM. DEFAULT=1024'
    )
    parser.add_argument(
        '--slm_height',
        type=int,
        default=768,
        help='Pixel height of SLM. DEFAULT=768'
    )
    parser.add_argument(
        '--segment_width',
        type=int,
        default=32,
        help='Pixel width of each segment (group of pixels on SLM). Must be a factor of the slm width. DEFAULT=32'
    )
    
    parser.add_argument(
        '--segment_height',
        type=int,
        default=24,
        help='Pixel height of each segment (group of pixels on SLM). Must be a factor of the slm height. DEFAULT=24'
    )

    parser.add_argument(
        '--num_masks',
        type=int,
        default=30,
        help='Initial population of randomly generated phase masks in genetic algorithm. DEFAULT=30'
    )
    parser.add_argument(
        '--num_childs',
        type=int,
        default=15,
        help='Number of offspring masks to generate each generation. DEFAULT=15'
    )
    parser.add_argument(
        '--gens',
        type=int,
        default=1000,
        help='Number of generations to run genetic algorithm. DEFAULT=1000'
    )
    parser.add_argument(
        '--mutate_initial_rate',
        type=float,
        default=.02,
        help='Initial mutation rate for genetic algorithm. DEFAULT=0.1'
    )
    parser.add_argument(
        '--mutate_final_rate',
        type=float,
        default=.001,
        help='Final mutation rate for genetic algorithm. DEFAULT=0.013'
    )
    parser.add_argument(
        '--mutate_decay_factor',
        type=float,
        default=650,
        help='Final mutation rate for genetic algorithm. DEFAULT=650'
    )
    parser.add_argument(
        '--num_phase_vals',
        type=int,
        default=256,
        help='Number of discrete phase values to be passed to SLM. DEFAULT=256'
    )
    parser.add_argument(
        '--fitness_func',
        type=str,
        default='max',
        help='Fitness function to use for ranking masks. OPTIONS: "mean", "max", "spot". DEFAULT="max"'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='oop_test',
        help='Path of text file to save optimized mask. DEFAULT="waveopt_output_files/"'
    )
    
    parser.add_argument(
        '--zernike_coeffs', nargs='*', type=int,
        default=[0],
        help='List of zernike coefficients for zernike modes 3-15. DEFAULT="0"'
    )
    parser.add_argument(
        '--grating_step', type=int,
        default=0,
        help='Blazed grating slope. DEFAULT="0"'
    )
    parser.add_argument(
        '--num_initial_metrics', type=int,
        default=500,
        help='Number of uniform mask measurements to average over for initial metric values. DEFAULT="100"'
    )

    main(parser.parse_args())
