import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, shutil

import Optimizer, Interface, Population


def main(args):
    start_num = '7-2_run1_newinterface'
    folder = '../run_'+str(start_num)
    os.makedirs(folder,exist_ok=True)
    shutil.copy('./take_zernike_data.py',folder+'/mainscript.py')
    shutil.copystat('./take_zernike_data.py',folder+'/mainscript.py')
    
    interface = Interface.Interface(args)


    args.num_initial_metrics = 500
    args.num_masks = 15
    args.num_childs = 15
    args.fitness_func = 'max'
    args0 = copy.copy(args)
    
    
    coeffs = [50,100,150,200]
    modes = np.arange(13)+3

##    segments = [[64,96],[64,48],[32,48],[32,24]]
    segments = [[64,96],[32,48]]

##    modes = [3]

    args = copy.copy(args0)
    args.save_path = folder+'/zopt'

    zopt = Optimizer.Optimizer(args,interface)
    zopt.run_zernike(modes,[-240,240])
    zopt_mask = zopt.parent_masks.get_slm_masks()[-1]
    
    for coeff in coeffs:
        for mode in modes:
            for segment in segments:
                for measure in [True]:
                    clist = np.zeros(13)
                    clist[mode-3]=coeff
                    args = copy.copy(args0)
                    args.zernike_coeffs = clist.tolist()

    ##                args.grating_step = 16
                    
    ##                args.segment_width = 64
    ##                args.segment_height = 48
                    args.segment_width = segment[0]
                    args.segment_height = segment[1]
                    args.gens = 1000
                    if segment[0]==32:
                        args.gens=1500
                    args.measure_all = measure

                    
                    segment_save = '/'+str(args.segment_height)+'_'+str(args.segment_width)
                    args.save_path = folder+'/mode_'+str(mode)+'_coeff_'+str(coeff) + segment_save + '_measure_'+str(measure)
                    
                    gopt = Optimizer.Optimizer(args,interface,base_mask=zopt_mask)
                    gopt.run_genetic()
                    print('\n\nDONE with genetic optimization............\n\n')

            print('\n\nDONE with zernike optimization............\n\n') 

            
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
        default=.01,
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
