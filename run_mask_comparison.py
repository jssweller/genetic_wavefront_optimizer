import numpy as np
from alerts import send_alert
import matplotlib.pyplot as plt
import time, datetime, sys, os, argparse, copy, shutil, traceback

import Optimizer, Interface, Population, textwrap

logfolder = r'Z:\118_data\delay_test'

def main(args):

    folders = [r'Z:\118_data\delay_test'
               ]
    
    start_time = [0,0,0] # [hour,minute,add days]
    run_time = [0,10,0] # [hours,minutes,seconds]

    interface = Interface.Interface(args)
    
    def compare(folder, start_time, run_time, interface):
        runid = '_compareall'
        run_description = 'Comparing performance of all masks in folder.'
        
        ### PARAMS ####
        numframes = 1
        zeromask = True

        zopt = Optimizer.Optimizer(args,interface)                

        # update zernike bestmask.txt for all zopt folders in dir.
        for root,dirs,files in os.walk(folder):
            zdirs = [os.path.join(root,d) for d in dirs if 'compare' not in d
                     and any([zz in d for zz in ['zopt','zspace']])]
            for zd in zdirs:
                zd+= '/optimized_zmodes.txt'
                if os.path.isfile(zd):
                    zopt.save_zernike_mask(zd)
                    print('UPDATED bestmask.txt: ', zd)

        # create or open compare_list.txt
        if not os.path.isfile(folder+'/compare_list.txt'):
            maskfiles = Optimizer.get_mask_compare_list(folder)
        else:
            f = open(folder+'/compare_list.txt')
            maskfiles = [x.strip() for x in f]
            f.close()
        for x in maskfiles:
            print(x)          
        
        zopt.run_compare_masks(start_time,
                              run_time,
                              numframes,
                              folder,
                              maskfiles,
                              runid,
                              run_description,
                              zeromask,
                              cmasks=None,
                              mask_labels=None)
        

    for folder in folders:
        compare(folder, start_time, run_time, interface)
        
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
        '--uniform_parent_prob',
        type=float,
        default=.1,
        help='Probability of choosing uniform mask as parent during breeding. DEFAULT=0.1'
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
        '--masktype',
        type=str,
        default='rect',
        help='Mask type to use for genetic algoritym. OPTIONS: "rect", "zernike"  DEFAULT= "rect" '
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
        '--zernike_modes', nargs='*', type=int,
        default=None,
        help='List of zernike modes to be used for genetic algorithm with zernike masks. DEFAULT=None'
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
    try:
        main(parser.parse_args())
    except Exception as e:
        print('caught error')
        print(traceback.format_exc())
        send_alert(message=str(e))
        try:
            with open(os.path.join(logfolder,'status.txt'),'a') as statusfile:
                statusfile.write(str(time.strftime('-----------------------\n\n')))
                statusfile.write(str(time.strftime("%a, %d %b %Y %H:%M:%S"))+'\n\n')
                statusfile.write(str(e)+'\n')
                statusfile.write(traceback.format_exc()+'\n')
        except Exception as a:
            send_alert(message=str(a))
        
    # run complete
    send_alert(message='', subject='Lab430 Run Ended.')
    try:
        with open(os.path.join(logfolder,'status.txt'), 'a') as statusfile:
            statusfile.write(str(time.strftime('-----------------------\n\n')))
            statusfile.write(str(time.strftime("%a, %d %b %Y %H:%M:%S"))+'\n\n')
            statusfile.write('Run Completed'+'\n')
    except Exception as a:
            send_alert(message=str(a)) 

