import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, __main__
import datetime as dt

import Interface, Population

class Optimizer:
    def __init__(self, args, interface, base_mask=0):
        self.args = copy.copy(args)
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
    
    def update_metrics(self, population=None, update_type='', save_mask=True):
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
            if save_mask==True:
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
            if save_mask==True:
                self.metrics['masks'].append(np.array(population.get_masks()[-1]).flatten())
        if update_type == 'final':
##            self.metrics['roi'][-1]=self.metrics['roi'][-2]
            if save_mask==True and len(self.metrics['masks'])>1:
                self.metrics['masks'][-1]=self.metrics['masks'][-2]

    def reset_time(self):
        self.time_start = time.time()
    
    def get_time(self):
        return datetime.timedelta(seconds=time.time()-self.time_start)
        
    def get_initial_metrics(self, save_mask=True):
        args0 = copy.copy(self.args)
        args0.zernike_coeffs = [0]
        args0.num_masks = 1
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        self.interface.get_output_fields(uniform_pop,repeat=self.num_masks_initial_metrics)
        self.update_metrics(uniform_pop, 'initial',save_mask=save_mask)
        os.makedirs(self.save_path, exist_ok=True)
        np.savetxt(self.save_path+'/initial_mean_intensity_roi.txt', uniform_pop.get_output_fields(), fmt='%d')
        
    def get_final_metrics(self):
        print('\nGetting final metrics...\n')
        args0 = copy.copy(self.args)
        self.parent_masks.ranksort()
        final_mask = np.array(self.parent_masks.get_slm_masks()[-1])
        masks = [final_mask,self.base_mask]
        mask_labels = ['final_mask','base_mask']
        zeromask = self.basemask != 0
        zopt.run_compare_masks(start_time=[0,0,0], run_time=[0,10,0], numframes=5, cmasks=masks, mask_labels=mask_labels)
        
    def get_final_mean_intensity(self):
        print('\nGetting final mean intensity...\n')
        args0 = copy.copy(self.args)
        args0.num_masks = 1
        args0.zernike_coeffs = [0]
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        self.interface.get_output_fields(uniform_pop,repeat=self.num_masks_initial_metrics)
        final_mean_intensity = uniform_pop.get_output_fields()
        np.savetxt(self.save_path+'/final_mean_intensity_roi.txt', final_mean_intensity, fmt='%d')
        file = open(self.save_path+'/log.txt','a')
        file.write('Final Avg Intensity: '+str(np.mean(final_mean_intensity))+'\n')
        file.close()

    def get_baseline_intensity(self, num_frames):
        print('Recording baseline intensity...')
        self.reset_time()
        args0 = copy.copy(self.args)
        args0.zernike_coeffs = [0]
        args0.num_masks = 1
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)

        self.interface.get_output_fields(uniform_pop,repeat=num_frames)
        print('.....Done\n')
        return uniform_pop.get_output_fields()

    def get_baseline_maxmean(self, run_minutes, num_to_avg):
        print('Recording baseline intensity...\n')
        self.reset_time()
        args0 = copy.copy(self.args)
        args0.zernike_coeffs = [0]
        args0.num_masks = 1
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)

        frame = 0
        times = []
        end_time = datetime.datetime.now() + datetime.timedelta(0,int(run_minutes*60))
        while datetime.datetime.now() < end_time:
            self.interface.get_output_fields(uniform_pop,repeat=num_to_avg)
            self.update_metrics(uniform_pop, 'initial',save_mask=False)
            times.append(datetime.datetime.now())
            frame += num_to_avg
            print('Time left:', end_time-datetime.datetime.now())

        self.save_path += '/baseline'
        os.makedirs(self.save_path, exist_ok=True)
        self.save_checkpoint()
        np.savetxt(self.save_path+'/baseline_times.txt', np.asarray(times,dtype='datetime64[s]'), fmt='%s')
        self.save_plots()

        self.save_path = args0.save_path
        print('.....Done\n\n')

    def save_rois_metrics(self, rois, population=None, save_path=None, logfile=None, append=False):
        self.init_metrics()
        if population is None:
            population = Population.Population(self.args,base_mask=self.base_mask,uniform=True)
        if save_path != None:
            self.save_path = save_path
        for field in rois:
            self.metrics['spot'].append(population.fitness(field,'spot'))
            self.metrics['maxint'].append(np.max(field))
            self.metrics['maxmet'].append(population.fitness(field,'max'))
            self.metrics['mean'].append(population.fitness(field,'mean'))
            self.metrics['roi'].append(field)
        self.save_checkpoint(append)
        self.save_plots()

        if logfile == None:
            f = self.save_path+'/averages.txt'
            file = open(f,'w+')
        else:
            file = logfile
        for key in self.metrics.keys():
            if key == 'roi':
                continue
            d = self.metrics[key]
            if len(d)>0:
                m = np.mean(d)
                if key == 'spot':
                    m = 1/m
                file.write('\n'+key+': '+str(m))
        file.write('\n\n')
        file.close()
                       
        self.save_path = self.args.save_path
    
    def run_generation(self):
        if self.gen==1:
            self.interface.get_output_fields(self.parent_masks)
            self.uniform_scale = np.mean([self.parent_masks.get_output_fields()[:2],self.parent_masks.get_output_fields()[-2:]])
            self.parent_masks.ranksort()                
            self.update_metrics()

        self.child_masks = self.parent_masks.make_children(self.uniform_childs)
        t0 = time.time()
        self.interface.get_output_fields(self.child_masks)
        tt = time.time() - t0
        
        if self.measure_all:
            self.child_masks.ranksort()
            t0 = time.time()
            self.interface.get_output_fields(self.parent_masks)
            tt += time.time()-t0
            self.parent_masks.ranksort()
        else:
##            if self.gen > int(self.numgens * 0.9):
##                self.interface.get_output_fields(self.parent_masks)
##                self.parent_masks.ranksort(scale=self.uniform_scale)
            self.child_masks.ranksort(scale=self.uniform_scale)

        print('SLMtime', round(tt,2),'s', end=' ...\t')
        self.parent_masks.replace_parents(self.child_masks)
        self.update_metrics()
        self.gen+=1
        

    def run_compare_masks(self,
                          start_time,
                          run_time,
                          numframes,
                          folder='',
                          maskfiles='',
                          runid='',
                          run_description='',
                          zeromask=False,
                          cmasks=None,
                          mask_labels=None):

        if folder == '':
            folder = self.save_path
        lenfolder = len(folder)
        folder = folder + '/compare_masks'+runid
        idnum = 0
        f = folder
        
        masks, labels, rois, times = [],[],[],[]
        nummasks = 0

        if cmasks == None:
            while os.path.isdir(folder):
                folder = f+str(idnum+1)
                idnum+=1
            os.makedirs(folder,exist_ok=True)
            file = open(folder+'/log.txt','w+')
            print('Run Description: ',run_description)
            file.write('Description: '+run_description+'\n\n')
            for mfile in maskfiles:
                if os.path.isfile(mfile):
                    file.write('Mask file:' + mfile + '\n')
            file.close()
            print('2')

            for mfile in maskfiles:
                if os.path.isfile(mfile):
                    genetic_mask = np.loadtxt(mfile, dtype=np.uint8).reshape([768,1024])
                    masks.append(genetic_mask)
                    labels.append(mfile[lenfolder:].replace('\\','--').replace('/','--').replace(':','yy'))
                    rois.append([])
                    times.append([])
                    nummasks+=1
                else:
                    print('File not found: ',mfile)

        else:
            for j, mask in enumerate(cmasks):
                masks.append(mask)
                if mask_labels==None:
                    labels.append('mask_'+str(j))
                else:
                    labels.append(mask_labels[j])
                

        print(labels)
        if zeromask:
            zero_mask = 0
            labels.append('nomask')
            rois.append([])
            times.append([])
            masks.append(zero_mask)
            nummasks+=1
        
        t0 = dt.datetime.now()
        start_time = dt.datetime.combine(t0.date() + dt.timedelta(days=start_time[2]),dt.time(hour=start_time[0], minute=start_time[1]))
        if dt.datetime.now() > start_time:
            start_time = dt.datetime.now()
        end_time = start_time + dt.timedelta(hours=run_time[0],minutes=run_time[1], seconds=run_time[2])

        while start_time > dt.datetime.now():
            print('WAITING... Time left before start:', start_time - dt.datetime.now())
            time.sleep(30)

        i=0
        while end_time > dt.datetime.now():
            num = i%nummasks
            print(labels[num], end='...')
            self.base_mask = masks[num]
            times[num].append(dt.datetime.now())
            rois[num].extend(self.get_baseline_intensity(numframes))
            i += 1
            print('Time left:',end_time - dt.datetime.now())
            
            if len(rois[num]) >= 1000:
                label = labels[num]
                fdir = folder+'/'+label
                os.makedirs(fdir,exist_ok=True)

                mode = 'w+'
                if os.path.isfile(folder+'/averages.txt') and num>0:
                    mode = 'a'
                file = open(folder+'/averages.txt',mode)
                file.write('\n\n'+label+' averaged: \n')
                self.save_rois_metrics(rois[num], save_path=fdir, logfile=file, append=True)
                np.savetxt(fdir+'/baseline_times.txt', np.asarray(times[num],dtype='datetime64[s]'), fmt='%s')
                rois[num]=[]
                times[num]=[]
            
        for i,label in enumerate(labels):
            fdir = folder+'/'+label
            os.makedirs(fdir,exist_ok=True)

            mode = 'w+'
            if os.path.isfile(folder+'/averages.txt') and i>0:
                mode = 'a'
            file = open(folder+'/averages.txt',mode)
            file.write('\n\n'+label+' averaged: \n')
            self.save_rois_metrics(rois[i], save_path=fdir, logfile=file, append=True)
            tfile = open(fdir+'/baseline_times.txt', 'a')
            np.savetxt(tfile, np.asarray(times[i],dtype='datetime64[s]'), fmt='%s')
            tfile.close()
    
    def run_genetic(self, numgens=None):
        if numgens is None:
            numgens = self.numgens
        print('genetic optimizer running...')
        self.gen=1
        self.reset_time()
        self.init_metrics()
        self.get_initial_metrics()
        self.initial_log()
        self.reset_time()
        while self.gen <= numgens:
            t0 = time.time()
            print('generation',self.gen,end=' ...\t')
            self.parent_masks.update_num_mutations(self.gen,numgens)
            self.run_generation()
            if self.gen % int(numgens/4)==0:
                self.save_checkpoint()
            print('Time', round(time.time()-t0,2),'s', end=' ...\t')
            print('Fitness:', round(max(self.parent_masks.get_fitness_vals()),2))

        self.get_final_metrics()
        self.save_checkpoint()
        self.final_log()
        self.save_plots()
        
    def run_zernike(self, zmodes, coeff_range):
        '''Zernike optimization algorithm returns best zernike coefficients in coeff_range'''
        print('Zernike optimizer running...')
        best_zmodes = np.zeros(26)
        self.args.zernike_coeffs = [0]
        self.init_metrics()
        args0 = copy.copy(self.args)
        args0.num_masks=1
##        args0.zernike_coeffs=[0]
        args0.fitness_func = 'max'
##        self.save_path=self.save_path+'/zernike'
        initial_base_mask = copy.copy(self.base_mask)
        base_mask = copy.copy(self.base_mask)
        self.parent_masks = Population.Population(args0,base_mask)
        self.get_initial_metrics(save_mask=False)
        self.initial_zernike_log(zmodes,coeff_range)
        
        for zmode in zmodes:
            if zmode<3 or zmode>26:
                print('Warning: Zernike mode number out of range (ignored).')
                continue
            print('\nOptimizing Zernike Mode',str(zmode))
            # Course search
            snum = 10
            coeffs = np.arange(coeff_range[0],coeff_range[1]+1,snum)
            best_coeff = self.get_best_coefficient(zmode,coeffs)
            
            # Fine Search
            coeffs = np.arange(best_coeff-snum,best_coeff+snum,2)
            best_coeff = self.get_best_coefficient(zmode,coeffs)

##            # Repeat Fine
##            for i in range(1):
##                coeffs = np.arange(best_coeff-int(snum/2),best_coeff+int(snum/2),2)
##                best_coeff = int(np.mean([self.get_best_coefficient(zmode,coeffs),best_coeff]))
            
            
            base_mask += self.parent_masks.create_zernike_mask(self.get_coeff_list(zmode,best_coeff))
            self.parent_masks.update_base_mask(base_mask)
            best_zmodes += self.get_coeff_list(zmode,best_coeff)
        
        np.savetxt(self.save_path+'/optimized_zmodes.txt',best_zmodes, fmt='%d')
        self.parent_masks.update_zernike_parent(best_zmodes)
        self.parent_masks.update_base_mask(initial_base_mask)
        self.interface.get_output_fields(self.parent_masks)
        self.update_metrics(save_mask=False)
        
        self.get_final_metrics()
        self.save_checkpoint()
        self.final_log()
        self.parent_masks.update_zernike_parent(best_zmodes)
        self.save_plots()
        self.save_path = args0.save_path

    def get_coeff_list(self,zmode,coeff):
        cfs = np.zeros(26)
        cfs[zmode-3] = coeff
        return cfs
    
    def get_best_coefficient(self,zmode,coeffs,repeat=30):
        maxmets=[]
        spotmets=[]
        print('coeff',end='')
        vals
        for i,coeff in enumerate(coeffs):
            print('...'+str(coeff),end='')
            self.parent_masks.update_zernike_parent(self.get_coeff_list(zmode,coeff))
##            if i==0 or coeffs[i-1]==0 or coeff==0:
##                self.parent_masks.update_zernike_parent(self.get_coeff_list(zmode,coeff))
##            else:
##                self.parent_masks.change_parent_zcoeff(coeff)
            self.parent_masks.update_zernike_parent(self.get_coeff_list(zmode,coeff))
            self.interface.get_output_fields(self.parent_masks,repeat)
            self.update_metrics(update_type='initial',save_mask=False)
            maxmets.append(self.metrics['maxmet'][-1])
            spotmets.append(self.metrics['spot'][-1])
        print('\n')
##        max_coeff = np.argmax(maxmets)
##        spot_coeff = np.argmin(spotmets)

        maxcoeff = Optimizer.get_polybest(coeffs,maxmets,np.argmax)
        spotcoeff = Optimizer.get_polybest(coeffs,spotmets,np.argmin)
        best_coeff = int((maxcoeff+spotcoeff)/2)
        
##        if isinstance(best_coeff,np.ndarray):
##            best_coeff = best_coeff[-1]
##        return coeffs[best_coeff]
        return best_coeff
 
    def initial_log(self):
        os.makedirs(self.save_path, exist_ok=True)
        file = open(self.save_path+'/log.txt','w+')
        file.write('Start Time: '+str(datetime.datetime.now()))
        file.write('\n\nMain script: '+str(os.path.realpath(__main__.__file__))+'\n\n')
        file.write('Save path: '+os.path.dirname(os.path.realpath(self.save_path+'/log.txt'))+'\n\n')
        file.write('#### Parameters ####:\n\n')

        file.write('Mode coefficients='+str(self.zernike_coeffs))
        for arg in vars(self.args):
            file.write('\n'+str(arg)+'='+str(getattr(self.args, arg)))
            
        file.write('\n\n#### Metrics ####:\n\n')
        file.write('Initial metrics time: '+str(self.get_time())+'\n')
        file.write('Initial Avg Intensity: '+str(self.metrics['mean'][0])+'\n')
        file.close()
    
    def initial_zernike_log(self,zmodes,coeff_range):
        os.makedirs(self.save_path, exist_ok=True)
        file = open(self.save_path+'/log.txt','w+')
        file.write('Start Time: '+str(datetime.datetime.now()))
        file.write('\n\nMain script: '+str(os.path.realpath(__main__.__file__))+'\n\n')
        file.write('Save path: '+os.path.dirname(os.path.realpath(self.save_path+'/log.txt'))+'\n\n')
        file.write('#### Parameters ####:\n\n')

        file.write('Zernike modes: '+str(zmodes)+'\n')
        file.write('Coefficient range: '+str(coeff_range))
        for arg in vars(self.args):
            file.write('\n'+str(arg)+'='+str(getattr(self.args, arg)))
            
        file.write('\n\n#### Metrics ####:\n\n')    
        file.write('Initial metrics time: '+str(self.get_time())+'\n')
        file.write('Initial Avg Intensity: '+str(self.metrics['mean'][0])+'\n')
        file.close()
    
    def final_log(self):
        file = open(self.save_path+'/log.txt','a')
        file.write('Final Spot Metric: '+str(1/self.metrics['spot'][-1])+'\n')
        file.write('Final Spot Enhancement: '+str(self.metrics['spot'][0]/self.metrics['spot'][-1])+'\n')          
        file.write('Final Intensity Enhancement: '+str(self.metrics['maxint'][-1]/self.metrics['maxint'][0])+'\n\n')
        file.write('Optimization Time: '+str(self.get_time())+'\n')
        file.write('end time: '+str(datetime.datetime.now())+'\n')
        file.close()
        
        
    def save_checkpoint(self, append=False):
        files = ['/spot_metric_vals_checkpoint.txt',
                 '/max_metric_vals_checkpoint.txt',
                 '/mean_intensity_vals_checkpoint.txt',
                 '/max_intensity_vals_checkpoint.txt',
                 '/roi.txt',
                 '/masks.txt']

        mode = 'w'
        if append: mode = 'a'
        
        f = []
        for file in files:
            f.append(open(self.save_path+file, mode))
            
        np.savetxt(f[0], 1/np.asarray(self.metrics['spot']), fmt='%10.3f')
        np.savetxt(f[1], self.metrics['maxmet'], fmt='%10.3f')
        np.savetxt(f[2], self.metrics['mean'], fmt='%10.3f')
        np.savetxt(f[3], self.metrics['maxint'], fmt='%d')
        np.savetxt(f[4], self.metrics['roi'], fmt='%d')
        np.savetxt(f[5],self.metrics['masks'], fmt='%d')
        np.savetxt(self.save_path+'/bestmask.txt',self.parent_masks.get_slm_masks()[-1], fmt = '%d')
        if not isinstance(self.parent_masks.get_base_mask(),int):
            np.savetxt(self.save_path+'/base_mask.txt',self.parent_masks.get_base_mask(), fmt = '%d')

        for x in f:
            x.close()
        
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
        plt.plot(1/np.asarray(self.metrics['spot']))
        plt.savefig(self.save_path+'/spot_metric_plot')
        plt.close()

        dim=int(np.sqrt(len(self.metrics['roi'][0])))
        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][-2]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/end_roi')
        plt.close()

        dim=int(np.sqrt(len(self.metrics['roi'][0])))
        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][-1]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/end_roi_averaged')
        plt.close()

        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][1]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/begin_roi')
        plt.close()
        
        plt.figure()
        plt.imshow(np.array(self.metrics['roi'][0]).reshape(dim,dim))
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/begin_roi_averaged')
        plt.close()

        plt.figure(figsize=(12,8))
        bmask = self.parent_masks.get_slm_masks()[-1]
        plt.imshow(bmask, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(self.save_path+'/final_mask')
        plt.close()

    def save_zernike_mask(self, zmodes_file):
        coeffs = np.loadtxt(zmodes_file)
        print(zmodes_file)
        print(coeffs.shape, coeffs)
        zmask = self.parent_masks.create_zernike_mask(coeffs)
        np.savetxt(os.path.dirname(zmodes_file)+'/bestmask.txt',zmask, fmt='%d')

####
    
        
def get_mask_compare_list(directory,names=['bestmask.txt'],write_to_file=True):
        maskfiles = []
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                f = os.path.join(root,d)
                for name in names:
                    mfile = os.path.join(f,name)
                    if os.path.isfile(mfile):
                        maskfiles.append(mfile)
                        print(mfile,'...added to compare list.')
        if write_to_file:
            with open(os.path.join(directory,'compare_list.txt'),'w+') as f:
                for m in maskfiles:
                    f.write(m+'\n')
        return maskfiles

def get_polybest(x,y,best_func=np.argmin):
    pfit = np.polyfit(x,y,2)
    p = np.poly1d(pfit)
    frange = np.arange(min(x),max(x),1)
    zbest = best_func(p(frange))
    return int(frange[zbest])
    
