import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse

class Interface:
    def __init__(self, args):
        self.num_bytes_buffer = args.bytes_buffer_size
        self.pipe_handle_in = wf.CreateFile(args.pipe_in_handle,
                     wf.GENERIC_READ | wf.GENERIC_WRITE,
                     0, None,
                     wf.OPEN_EXISTING,
                     0,None)
                                 
        self.pipe_handle_out = wf.CreateFile(args.pipe_out_handle,
                     wf.GENERIC_READ | wf.GENERIC_WRITE,
                     0, None,
                     wf.OPEN_EXISTING,
                     0,None)
        
    def get_pipe_vals():
        return self.pipe_handle_in, self.pipe_handle_out, self.num_bytes_buffer

    def get_buffer_size(self):
        """Return buffer size as int."""
        buffer_size = int.from_bytes(wf.ReadFile(self.pipe_handle_in, self.num_bytes_buffer)[1], byteorder='little', signed=False)
        return buffer_size

    def flatten_mask(self,mask):
        """Flatten mask. Return 1d array."""
        return mask.flatten()
    
    def encode_mask(self,mask):
        """Return buffer size and mask as bytearray."""
        return bytearray(len(mask).to_bytes(self.num_bytes_buffer, byteorder='little', signed=False)) + bytearray(mask)
    
    def get_output_fields(self,population):
        """Transmit mask pixel data through pipe to apparatus. Return list of output field ROIs."""      
        input_masks = population.get_slm_masks()
        t0=time.time()
        
        roi_list = []
##        print(np.shape(input_masks))
##        print('sending...',end='')
        for i,mask in enumerate(input_masks):
##            print(str(i)+'..',end='')
            
            mask = self.flatten_mask(mask).astype(np.uint8)
            wf.WriteFile(self.pipe_handle_out, self.encode_mask(mask))
            read_pipe = wf.ReadFile(self.pipe_handle_in, self.get_buffer_size())
            read_array = list(read_pipe[1])
            roi_list.append(read_array[0::3])
##        print('\nInterface Time (seconds): ', time.time()-t0)
        population.update_output_fields(roi_list)
        
    
