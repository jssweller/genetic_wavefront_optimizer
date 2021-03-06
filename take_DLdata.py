import numpy as np
from alerts import send_alert
import matplotlib.pyplot as plt
import time, sys, os, argparse, copy, shutil

import Optimizer, Interface, Population, textwrap

start_num = 'DLdata_c128_z3-35_112x112_1m_200_centered'
folder = 'Z:/118_data/DL_data/'+start_num


def main(args,folder):
    overwrite_data = False
    
    run_description = 'Simulated aberrations and images for Deep Learning Training \n\
Lab 118. \n\
112x112 image dimensions. \n\
Zernike Polynomials 1-35. \n\
Fully randomized aberration generation. \n\
Zernike optimization using corrected scaling. \n\
zbasis = True. \n\
Exposure value at -6.'
    os.makedirs(folder,exist_ok=True)
    shutil.copy(sys.argv[0],folder+'/mainscript.py')
    shutil.copystat(sys.argv[0],folder+'/mainscript.py')

    file = open(folder+'/log.txt','w+')
    print('Run Description: ',run_description)
    file.write('Description: '+run_description+'\n\n')
    file.close()
    interface = Interface.Interface(args)

    args.num_initial_metrics = 10
    args0 = copy.copy(args)
    
    
    optimize_zernike = True
   
    zopt_mask = 0
    if optimize_zernike:
        args = copy.copy(args0)
##        args.save_path = folder+'/zopt'
        args.save_path = folder[:folder.rfind('DL_data')+len('DL_data')]
        zmodes = np.arange(3,49)
        zopt = Optimizer.Optimizer(args,interface)
        if os.path.isfile(args.save_path+'/optimized_zmodes.txt'):
            print('Loading zmodes from file...')
            opt_zmodes = np.loadtxt(args.save_path+'/optimized_zmodes.txt')
            print(opt_zmodes)
            zopt_mask = zopt.parent_masks.create_zernike_mask(opt_zmodes)
            print(zopt_mask.shape)
        else:
            args.save_path = folder+'/zopt'
            zopt.run_zernike(zmodes,[-600,600])
            zopt_mask = zopt.parent_masks.get_slm_masks()[-1]
    
    coeff_range = [-128,128]
    DLmodes = np.arange(1,36)
    num_data = 1000000
    batch_size = 1000

    args = copy.copy(args0)
    args.save_path = folder+'/DLdata'
    DLopt = Optimizer.Optimizer(args,interface,base_mask=zopt_mask)
    DLopt.record_DLdata(DLmodes, coeff_range, num_data, batch_size, overwrite=overwrite_data)
                
                 
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
        default=50,
        help='Number of uniform mask measurements to average over for initial metric values. DEFAULT="100"'
    )

    main_error = False    
    try:
        main(parser.parse_args(), folder)
    except Exception as e:
        print('caught error')
        main_error = True
        send_alert(message=str(e))
        try:
            with open(os.path.join(folder,'status.txt'),'a') as statusfile:
                statusfile.write(str(time.strftime('-----------------------\n\n')))
                statusfile.write(str(time.strftime("%a, %d %b %Y %H:%M:%S"))+'\n\n')
                statusfile.write(str(e)+'\n')
                statusfile.write(traceback.format_exc()+'\n')
                print(traceback.format_exc())
        except Exception as a:
            send_alert(message=str(a))
        
    # run complete
    if not main_error:
        send_alert(message='Lab 118 Run ended without error.', subject='Lab 118 Run Completed Without Error.')
        try:
            with open(os.path.join(folder,'status.txt'), 'a') as statusfile:
                statusfile.write(str(time.strftime('-----------------------\n\n')))
                statusfile.write(str(time.strftime("%a, %d %b %Y %H:%M:%S"))+'\n\n')
                statusfile.write('Run Completed'+'\n')
        except Exception as a:
                send_alert(message=str(a)) 

