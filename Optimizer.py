import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse

import Interface, Population

class Optimizer:
    def __init__(self, args, interface, base_mask=0):
        self.args = args
        self.base_mask = base_mask
        self.zernike_coeffs = args.zernike_coeffs # list of zernike coefficients
        self.grating_step = args.grating_step
        self.save_path = args.save_path
        self.numgens = args.gens
        self.uniform_childs = args.add_uniform_childs
        self.num_masks_initial_metrics = args.num_initial_metrics
        self.measure_all = args.measure_all
        
        self.gen = 0
        
        
        self.interface = interface
        self.parent_masks = Population.Population(self.args,base_mask)
        self.child_masks = 0
        
        self.init_metrics()
        self.reset_time()
        
    def init_metrics(self):
        self.metrics={'masks':[], 'roi':[], 'maxint':[], 'spot':[],'maxmet':[],'mean':[]}
    
    def update_metrics(self, population=None, update_type=''):
        if (update_type == 'final' or update_type == 'initial') and not (population is None):
            spot_metrics, mean_metrics, max_metrics, = [],[],[]
            for field in population.get_output_fields():
                spot_metrics.append(population.fitness(field,'spot'))
                mean_metrics.append(population.fitness(field,'mean'))
                max_metrics.append(population.fitness(field,'max'))
            roi = population.get_output_fields()  
            masks = population.get_masks()
            self.metrics['spot'].append(np.mean(spot_metrics))
            self.metrics['maxint'].append(np.mean(np.max(roi,axis=1)))
            self.metrics['maxmet'].append(np.mean(max_metrics))
            self.metrics['mean'].append(np.mean(mean_metrics))
            self.metrics['roi'].append(np.mean(roi,axis=0))
            self.metrics['masks'].append(np.array(np.mean(masks,axis=0)).flatten())
        else:
            population = self.parent_masks
            population.ranksort()
            field = population.get_output_fields()[-1]
            self.metrics['spot'].append(population.fitness(field,'spot'))
            self.metrics['maxint'].append(np.max(field))
            self.metrics['maxmet'].append(population.fitness(field,'max'))
            self.metrics['mean'].append(population.fitness(field,'mean'))
            self.metrics['roi'].append(field)
            self.metrics['masks'].append(np.array(population.get_masks()[-1]).flatten())
        if update_type == 'final':
            self.metrics['roi'][-1]=self.metrics['roi'][-2]
            self.metrics['masks'][-1]=self.metrics['masks'][-2]

    def reset_time(self):
        self.time_start = time.time()
    
    def get_time(self):
        return datetime.timedelta(seconds=time.time()-self.time_start)
        
    def get_initial_metrics(self):
        args0 = self.args
        args0.num_masks = self.num_masks_initial_metrics
        args0.zernike_coeffs = [0]
##        print('initial metric masks ',args0.num_masks)
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        self.interface.get_output_fields(uniform_pop)
        self.update_metrics(uniform_pop, 'initial')
        initial_mean_intensity = self.parent_masks.get_output_fields()
        np.savetxt(self.save_path+'/initial_mean_intensity_roi.txt', initial_mean_intensity, fmt='%d')
        
    def get_final_metrics(self):
        args0 = self.args
##        args0.num_masks = self.num_masks_initial_metrics
        self.parent_masks.ranksort()
        final_mask = self.parent_masks.get_masks()[-1]
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        uniform_pop.update_masks([final_mask]*self.num_masks_initial_metrics)
        self.interface.get_output_fields(uniform_pop)
        self.update_metrics(uniform_pop, 'final')
        self.get_final_mean_intensity()
        
    def get_final_mean_intensity(self):
        args0 = self.args
        args0.num_masks = self.num_masks_initial_metrics
        args0.zernike_coeffs = [0]
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        self.interface.get_output_fields(uniform_pop)
        final_mean_intensity = self.parent_masks.get_output_fields()
        np.savetxt(self.save_path+'/final_mean_intensity_roi.txt', final_mean_intensity, fmt='%d')
        file = open(self.save_path+'/log.txt','a')
        file.write('\n Final Avg Intensity: '+str(np.mean(final_mean_intensity)))
        file.close()
    
    
    def run_generation(self):
        if self.gen==1:
            self.interface.get_output_fields(self.parent_masks)
            self.parent_masks.ranksort()
            self.update_metrics()
            
        self.child_masks = self.parent_masks.make_children(self.uniform_childs)
        self.interface.get_output_fields(self.child_masks)
        self.child_masks.ranksort()
        
        if self.measure_all:
            self.interface.get_output_fields(self.parent_masks)
            self.parent_masks.ranksort()
        
        self.parent_masks.replace_parents(self.child_masks)
        self.update_metrics()
        self.gen+=1
    
    def run_genetic(self, numgens=None):
        if numgens is None:
            numgens = self.numgens
        print('genetic optimizer running...')
        self.gen=1
        self.reset_time()
        self.parent_masks.init_zernike_mask()
        self.init_metrics()
        self.get_initial_metrics()
        self.initial_log()
        
        self.reset_time()
        self.parent_masks.init_masks()
        while self.gen <= numgens:
            print('generation',self.gen,' ....',end='\t')
            self.parent_masks.update_num_mutations(self.gen,numgens)
            self.run_generation()
            if self.gen % int(numgens/4)==0:
                self.save_checkpoint()
            print('Fitness:', round(max(self.parent_masks.get_fitness_vals()),2))

        self.get_final_metrics()
        self.save_checkpoint()
        self.final_log()
        self.save_plots()
        
    def run_zernike(self, zmodes, coeff_range):
        '''Zernike optimization algorithm returns best zernike coefficients in coeff_range'''
        best_zmodes = np.zeros(13)
        self.parent_masks.init_zernike_mask()
        self.init_metrics()
        args0 = self.args
        args0.pop=1
        args0.fitness_func = 'spot'
        self.save_path=self.save_path+'/zernike'
        initial_base_mask = self.base_mask
        base_mask = self.base_mask
        self.parent_masks = Population.Population(args0,base_mask)
        self.parent_masks.init_zernike_mask()
        
        self.initial_zernike_log(zmodes,coeff_range)
        
        for zmode in zmodes:
            if zmode<3 or zmode>15:
                print('Warning: Zernike mode number out of range (ignored).')
                continue
            # Course search
            snum = 10
            coeffs = np.arange(coeff_range[0],coeff_range[1],snum)
            best_coeff = self.get_best_coefficient(zmode,coeffs)
            
            # Fine Search
            coeffs = np.arange(best_coeff-snum,best_coeff+snum,1)
            best_coeff = self.get_best_coefficient(zmode,coeffs)
            
            base_mask += self.parent_masks.create_zernike_mask(self.get_coeff_list(zmode,best_coeff))
            self.parent_masks.update_base_mask(base_mask)
            best_zmodes += self.get_coeff_list(zmode,best_coeff)
        
        np.savetxt(self.save_path+'/zmodes.txt',best_zmodes, fmt='%d')
        self.parent_masks.update_masks([self.parent_masks.create_mask(True)])
        
        self.base_mask = initial_base_mask
        self.get_final_metrics()
        self.save_checkpoint()
        self.final_log()
        self.save_plots()
        self.save_path = args0.save_path
        
    def get_coeff_list(self,zmode,coeff):
        cfs = np.zeros(13)
        cfs[zmode-3] = coeff
        return cfs
    
    def get_best_coefficient(self,zmode,coeffs):
        best_coeff=coeffs[0]
        for i,coeff in enumerate(coeffs):
            self.parent_masks.update_zernike_parent(self.get_coeff_list(zmode,coeff))
            self.interface.get_output_fields(self.parent_masks)
            self.update_metrics()
            if i>0:
                if self.metrics['spot'][-1]>self.metrics['spot'][-2]:
                    best_coeff=coeff
            else:
                best_coeff=coeff
        return best_coeff
    
    def initial_log(self):
        os.makedirs(self.save_path, exist_ok=True)
        file = open(self.save_path+'/log.txt','w+')
        file.write('This is the log file for wave_opt.py.\n\n')
        file.write('Run_name: '+self.save_path+'\n\n')
        file.write('Parameters:\n')

        file.write('\nzernike mode='+str(self.zernike_coeffs))
        file.write('\nmode coefficient='+str(self.zernike_coeffs))
        for arg in vars(self.args):
            file.write('\n'+str(arg)+'='+str(getattr(self.args, arg)))

        file.write('\n\ninitial metrics time: '+str(self.get_time()))
        file.close()
    
    def initial_zernike_log(self,zmodes,coeff_range):
        os.makedirs(self.save_path, exist_ok=True)
        file = open(self.save_path+'/log.txt','w+')
        file.write('This is the zernike log file for wave_opt.py.\n\n')
        file.write('Run_name: '+self.save_path+'\n\n')
        file.write('Parameters:\n')
        file.write('\nzernike modes: '+str(zmodes))
        file.write('\ncoefficient range: '+str(coeff_range))
        for arg in vars(self.args):
            file.write('\n'+str(arg)+'='+str(getattr(self.args, arg)))
            
        file.write('\n\ninitial metrics time: '+str(self.get_time()))
        file.close()
    
    def final_log(self):
        file = open(self.save_path+'/log.txt','a')
        file.write('\nFinal Spot Metric: '+str(self.metrics['spot'][-1]))
        file.write('\nFinal Spot Enhancement: '+str(self.metrics['spot'][-1]/self.metrics['spot'][0]))          
        file.write('\nFinal Intensity Enhancement: '+str(self.metrics['maxint'][-1]/self.metrics['maxint'][0]))
        file.write('\n\nOptimization Time: '+str(self.get_time()))
        file.close()
        
        
    def save_checkpoint(self):
        np.savetxt(self.save_path+'/spot_metric_vals_checkpoint.txt', self.metrics['spot'], fmt='%10.3f')
        np.savetxt(self.save_path+'/max_metric_vals_checkpoint.txt', self.metrics['maxmet'], fmt='%10.3f')
        np.savetxt(self.save_path+'/mean_intensity_vals_checkpoint.txt', self.metrics['mean'], fmt='%10.3f')
        np.savetxt(self.save_path+'/max_intensity_vals_checkpoint.txt', self.metrics['maxint'], fmt='%d')
        np.savetxt(self.save_path+'/roi.txt', self.metrics['roi'], fmt='%d')
        np.savetxt(self.save_path+'/masks.txt',self.metrics['masks'], fmt='%d')
        if not isinstance(self.parent_masks.get_base_mask(),int):
            np.savetxt(self.save_path+'/base_mask.txt',self.parent_masks.get_base_mask(), fmt = '%d')
        
    def save_plots(self):
        plt.figure()
        plt.plot(self.metrics['maxmet'])
        plt.savefig(self.save_path+'/max_metric_plot')
        plt.close()

        plt.figure()
        plt.plot(self.metrics['maxint'])
        plt.savefig(self.save_path+'/max_intensity_plot')
        plt.close()

        plt.figure()
        plt.plot(self.metrics['mean'])
        plt.savefig(self.save_path+'/mean_intensity_plot')
        plt.close()

        plt.figure()
        plt.plot(self.metrics['spot'])
        plt.savefig(self.save_path+'/spot_metrics_plot')
        plt.close()

        dim=int(np.sqrt(len(self.metrics['roi'][0])))
        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][-1]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/end_roi')
        plt.close()
        
        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][0]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/begin_roi')
        plt.close()

        plt.figure(figsize=(12,8))
        bmask = self.parent_masks.get_slm_masks()[-1]
        plt.imshow(bmask, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/final_mask')
        plt.close()
        
    def grab_screenshot(self,name):
        ImageGrab.grab().save(self.save_path+'/'+name+'_screenshot.png')
        

