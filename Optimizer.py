import matplotlib
matplotlib.use('Agg') # Added to fix RuntimeError in tkinter when saving plot images
from matplotlib import pyplot as plt

import numpy as np
import time, datetime, sys, os, argparse, copy, __main__
from scipy.signal import medfilt

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
    
    def update_metrics(self, population=None, update_type='', save_mask=True, save_roi=True):
        if population is None:
            population = self.parent_masks
            
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
            if save_mask:
                self.metrics['masks'].append(np.array(np.mean(masks,axis=0)).flatten())
            if save_roi:
                self.metrics['roi'].append(np.mean(roi,axis=0))

        elif update_type == 'roi_only':
            roi = population.get_output_fields()
            self.metrics['roi'] += roi

        else:
            population = self.parent_masks
            population.ranksort()
            field = population.get_output_fields()[-1]
            self.metrics['spot'].append(population.fitness(field,'spot'))
            self.metrics['maxint'].append(np.max(field))
            self.metrics['maxmet'].append(population.fitness(field,'max'))
            self.metrics['mean'].append(population.fitness(field,'mean'))
            if save_mask==True:
                self.metrics['masks'].append(np.array(population.get_masks()[-1]).flatten())
            if save_roi:
                self.metrics['roi'].append(field)

        if update_type == 'final':
            if save_mask==True and len(self.metrics['masks'])>1:
                self.metrics['masks'][-1]=self.metrics['masks'][-2]

    def reset_time(self):
        self.time_start = time.time()
    
    def get_time(self):
        return datetime.timedelta(seconds=time.time()-self.time_start)
        
    def get_initial_metrics(self, save_mask=True, save_roi=True):
        print('Getting initial metrics...')
        args0 = copy.copy(self.args)
        args0.zernike_coeffs = [0]
        args0.num_masks = 1
        args0.masktype = 'rect'
        uniform_pop = Population.Population(args0,base_mask=self.base_mask,uniform=True)
        self.interface.get_output_fields(uniform_pop,repeat=self.num_masks_initial_metrics)
        self.update_metrics(uniform_pop, 'initial',save_mask=save_mask, save_roi=save_roi)
        os.makedirs(self.save_path, exist_ok=True)
        np.savetxt(self.save_path+'/initial_mean_intensity_roi.txt', uniform_pop.get_output_fields(), fmt='%d')
        print('...done')
        
    def get_final_metrics(self):
        print('\nGetting final metrics...\n')
        args0 = copy.copy(self.args)
        self.parent_masks.ranksort()
        final_mask = np.array(self.parent_masks.get_slm_masks()[-1])
        masks = [final_mask,self.base_mask]
        mask_labels = ['final_mask','base_mask']
        zeromask = self.base_mask != 0
        self.run_compare_masks(start_time=[0,0,0], run_time=[0,10,0], numframes=5, cmasks=masks, mask_labels=mask_labels)
        
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

        if len(rois) > 0:
            for field in rois:
                self.metrics['spot'].append(population.fitness(field,'spot'))
                self.metrics['maxint'].append(np.max(field))
                self.metrics['maxmet'].append(population.fitness(field,'max'))
                self.metrics['mean'].append(population.fitness(field,'mean'))
                self.metrics['roi'].append(field)
            self.save_checkpoint(append)
        self.load_checkpoint(load_masks=False)
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

        self.init_metrics()                       
        self.save_path = self.args.save_path
    

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
                nummasks += 1
                rois.append([])
                times.append([])
                if mask_labels==None:
                    labels.append('mask_'+str(j))
                else:
                    labels.append(mask_labels[j])
                

        print(''.join([x+'\n' for x in labels]))
        if zeromask:
            zero_mask = 0
            labels.append('nomask')
            rois.append([])
            times.append([])
            masks.append(zero_mask)
            nummasks+=1
        
        t0 = datetime.datetime.now()
        start_time = datetime.datetime.combine(t0.date() + datetime.timedelta(days=start_time[2]),datetime.time(hour=start_time[0], minute=start_time[1]))
        if datetime.datetime.now() > start_time:
            start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(hours=run_time[0],minutes=run_time[1], seconds=run_time[2])

        while start_time > datetime.datetime.now():
            print('WAITING... Time left before start:', start_time - datetime.datetime.now())
            time.sleep(30)

        masknums = np.arange(0,nummasks)
        while end_time > datetime.datetime.now():
            np.random.shuffle(masknums)
            for num in masknums:
                print(labels[num], end='...')
                self.base_mask = masks[num]
                times[num].append(datetime.datetime.now())
                rois[num].extend(self.get_baseline_intensity(numframes))
                print('Time left:',end_time - datetime.datetime.now())
                
                if len(rois[num]) >= 100:
                    label = labels[num]
                    fdir = folder+'/'+label
                    os.makedirs(fdir,exist_ok=True)

                    mode = 'w+'
                    if os.path.isfile(folder+'/averages.txt') and num>0:
                        mode = 'a'
                    file = open(folder+'/averages.txt',mode)
                    file.write('\n\n'+label+' averaged: \n')
                    self.save_rois_metrics(rois[num], save_path=fdir, logfile=file, append=True)
                    tfile = open(fdir+'/baseline_times.txt', 'a')
                    np.savetxt(tfile, np.asarray(times[num],dtype='datetime64[s]'), fmt='%s')
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
        

    def run_compare_all_in_folder(self,folder,run_time):
        ### PARAMS ####
        runid = '_compareall'
        run_description = 'Comparing performance of all masks in folder.'
        start_time = [0,0,0] # [hour,minute,add days]
        numframes = 1
        zeromask = True
        
        if not os.path.isfile(folder+'/compare_list.txt'):
            maskfiles = get_mask_compare_list(folder)
        else:
            maskfiles = np.loadtxt(folder+'/compare_list.txt',dtype=np.str)
        print(''.join([x+'\n' for x in maskfiles]))
        self.run_compare_masks(start_time,
                          run_time,
                          numframes,
                          folder,
                          maskfiles,
                          runid,
                          run_description,
                          zeromask,
                          cmasks=None,
                          mask_labels=None)
        
    
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
            self.child_masks.ranksort(scale=self.uniform_scale)

        print('SLMtime', round(tt,2),'s', end=' ...\t')
        self.parent_masks.replace_parents(self.child_masks)
        self.update_metrics()
        self.gen+=1
    
    def run_genetic(self, numgens=None):
        if numgens is None:
            numgens = self.numgens
        print('genetic optimizer running...')
        self.gen=1

        # check for previous run data
        if os.path.isfile(os.path.join(self.save_path,'max_intensity_vals_checkpoint.txt')):
            print('Previous run data found. Checking if completed...')
            self.load_checkpoint(os.path.join(self.save_path,''))
            gen = len(self.metrics['maxint'])
            if gen < numgens:
                print('Previous run not completed. Continuing...')
                self.gen = gen - 1
                self.parent_masks.load_masks(self.save_path, name='masks_parent_checkpoint')
            else:
                print('This run has already completed! Aborting current run...')
                return

        else:
            # if no previous run data, start normally
            self.reset_time()
            self.init_metrics()
            print('Get initial metrics...', end='')
            self.get_initial_metrics()
            print('...done')
            self.initial_log()
            self.reset_time()
                                 
                      
        while self.gen <= numgens:
            t0 = time.time()
            print('generation',self.gen,end=' ...\t')
            self.parent_masks.update_num_mutations(self.gen,numgens)
            self.run_generation()
            if self.gen % min(int(numgens/min(4,numgens)),100) == 0:
                self.save_checkpoint()
                self.save_plots()
                self.parent_masks.save_masks(directory = self.save_path, name='masks_parent_checkpoint')
            print('Time', round(time.time()-t0,2),'s', end=' ...\t')
            fval = max(self.parent_masks.get_fitness_vals())
            if fval < 1:
                print('Fitness: %.2e' % (fval))
            else:
                print('Fitness: %.2f' % (fval))

        self.save_checkpoint()
        self.final_log()
        self.save_plots()
        self.get_final_metrics()

    def run_zernike(self, zmodes, coeff_range, num_runs=2, cumulative=True):
        '''Zernike optimization algorithm returns best zernike coefficients in coeff_range'''
        print('Zernike optimizer running...')
        self.args.zernike_coeffs = np.zeros(49)
        initial_base_mask = copy.copy(self.base_mask)
        intial_save_path = self.save_path

        run_start = 0
        os.makedirs(self.save_path, exist_ok=True)
        for d in sorted(next(os.walk(self.save_path))[1]):
            if 'run' in d and os.path.isfile(os.path.join(self.save_path,d,'optimized_zmodes.txt')):
                run_start = int(d.split('_')[1]) + 1

        print('run start = ', run_start)
        
        for run in range(run_start,num_runs):
            self.init_metrics()
            args0 = copy.copy(self.args)
            args0.num_masks = 1
            args0.fitness_func = 'spot'
            
            base_mask = initial_base_mask

            zmodes_file = os.path.join(intial_save_path,'run_'+str(run-1),'optimized_zmodes.txt')
            if os.path.isfile(zmodes_file):
                args0.zernike_coeffs = np.loadtxt(zmodes_file)
                print('loading optimized_zmodes', zmodes_file)
                print(args0.zernike_coeffs)

            self.save_path = os.path.join(intial_save_path,'run_'+str(run))
            os.makedirs(self.save_path, exist_ok=True)
            
            self.parent_masks = Population.Population(args0,base_mask)
            self.get_initial_metrics(save_mask=False)
            self.initial_zernike_log(zmodes,coeff_range)
            

            best_zmodes = np.zeros(49)
            for zmode in zmodes:
                if zmode<3 or zmode>=49:
                    print('Warning: Zernike mode number out of range(ignored). Number: '+str(zmode))
                    continue
                print('\nOptimizing Zernike Mode',str(zmode))

                if self.parent_masks.zernike_coeffs[zmode] != 0:
                    self.parent_masks.zernike_coeffs[zmode] = 0
                    print(self.parent_masks.zernike_coeffs)

                self.parent_masks.init_zernike_mask()
                # Course search
                snum = 10
                coeffs = np.arange(coeff_range[0],coeff_range[1]+1,snum)
                best_coeff, next_coeffs = self.get_best_coefficient(zmode, coeffs, method='argmax', metric='spot', shuffle=True)
                print('next_coeffs:', next_coeffs)
                
                # Fine Search
##                snum = 4
##                coeffs = np.arange(next_coeffs[0], next_coeffs[1]+1, snum)
                num_coeffs = 100
                coeffs = np.linspace(next_coeffs[0], next_coeffs[1]+1, num_coeffs, dtype=int)
                best_coeff, next_coeffs = self.get_best_coefficient(zmode, coeffs, method='poly', metric='spot', repeat=1, shuffle=True)
            
                if cumulative:
                    self.parent_masks.zernike_coeffs[zmode] = best_coeff

                best_zmodes += self.get_coeff_list(zmode,best_coeff)
                self.save_checkpoint()
                self.save_plots()
                
            np.savetxt(os.path.join(self.save_path,'optimized_zmodes.txt'),best_zmodes , fmt='%d')
            self.parent_masks.init_zernike_mask()
            self.parent_masks.update_zernike_parents(np.zeros(49))
            self.parent_masks.update_base_mask(initial_base_mask)
            self.interface.get_output_fields(self.parent_masks)
            self.update_metrics(save_mask=True)
            
            self.save_checkpoint()
            self.final_log()
            self.save_plots()
            self.get_final_metrics()
            self.save_path = args0.save_path

    def map_zspace(self, zmodes, coeff_range, repeat=10, cumulative=True):
        '''Zernike optimization algorithm returns best zernike coefficients in coeff_range'''
        print('Zernike optimizer running...')
        best_zmodes = np.zeros(49)
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
##        self.get_initial_metrics(save_mask=False)
        self.initial_zernike_log(zmodes,coeff_range)
        
        for zmode in zmodes:
            if zmode<3 or zmode>=49:
                print('Warning: Zernike mode number out of range(ignored). Number: '+str(zmode))
                continue
            print('\nOptimizing Zernike Mode',str(zmode))
            self.init_metrics()
            self.save_path = os.path.join(args0.save_path,'mode_'+str(zmode))
            os.makedirs(self.save_path,exist_ok=True)
            
            # Course search
            snum = 1
            coeffs = np.arange(coeff_range[0],coeff_range[1]+1,snum)
            best_coeff = self.get_best_coefficient(zmode, coeffs, repeat=repeat, record_all_data=True, shuffle=True)
        
            if cumulative:
                base_mask += self.parent_masks.create_zernike_mask(self.get_coeff_list(zmode,best_coeff))
            self.parent_masks.update_base_mask(base_mask)
            best_zmodes += self.get_coeff_list(zmode,best_coeff)
            self.save_checkpoint()
            self.save_plots()

        self.save_path = args0.save_path
        os.makedirs(self.save_path,exist_ok=True)
        np.savetxt(self.save_path+'/optimized_zmodes.txt',best_zmodes, fmt='%d')
        coeff_vector = np.asarray([[x]*repeat for x in coeffs]).flatten()
        np.savetxt(self.save_path+'/coeff_vector.txt',coeff_vector, fmt='%d')
        self.save_path = args0.save_path


    def record_DLdata(self, zmodes, coeff_range, num_data, batch_size=1000, overwrite=True):
        '''Randomly loads zernike aberrations to SLM and records coefficient vector and ROI'''
        print('Recording Deep Learning data...')
        zlist = []
        self.init_metrics()
        self.args.zernike_coeffs=[0]
        self.get_initial_metrics(save_mask=False)
        self.save_plots()
        
        args0 = copy.copy(self.args)
        args0.num_masks=1
        self.parent_masks = Population.Population(args0,self.base_mask, uniform=True)
        self.init_metrics()
        self.initial_zernike_log(zmodes,coeff_range)
        
        zmodes = np.arange(max(1,min(zmodes)),min(49,max(zmodes)))
        coeff_range = np.arange(coeff_range[0],coeff_range[1]+1)

        numps = np.arange(0,max(zmodes.shape)+1)
        p = (numps+0.1)**(1/1.5) # probability dist for choosing number of polynomials
        p = p/sum(p)
        
        t0 = time.time()
        tbatch = time.time()

        i=0
        send_to_slm = 100

        if not overwrite:
            print('Checking for existing data to avoid overwriting')
            folders = sorted(next(os.walk(self.save_path))[1], key= lambda x: int(x[x.rfind('data')+4:]))
            if len(folders)>0:
                batch_size = int(folders[0][folders[0].rfind('data')+4:])
                if len(folders) > 1:
                    batch_size = int(folders[1][folders[1].rfind('data')+4:]) - batch_size
                print('Using batch_size of previous data:', batch_size)
                for d in folders:
                    if os.path.isfile(os.path.join(self.save_path,d,'zcoeffs.txt')):
                        i = int(d[d.rfind('data')+4:])
                
        print('starting with i =',i)
        while i <= num_data:
            num_send = min(send_to_slm, batch_size - (i % batch_size))
            i += num_send
            for b in range(num_send):
                c_numps = np.random.choice(numps,size=1, p=p) # randomly choose number of polynomials in aberration
                c_zmodes = np.random.choice(zmodes, c_numps, replace=False) # choose polynomials
                coeffs = np.random.choice(coeff_range, c_numps, replace=True) # choose coefficients
                clist = self.get_coeff_list(c_zmodes,coeffs) # get formatted coefficient list
                zlist.append(clist)

            self.parent_masks.update_zernike_parents(zlist[-num_send:])
            self.interface.get_output_fields(self.parent_masks)
            self.update_metrics(update_type='roi_only', save_mask=False)

            if  i % batch_size==0 or i==num_data:
                print('\nBatch '+str(int((i+1)/batch_size)), '\nTotal Time:\t\t',time.time()-t0)
                print('Batch Time:\t\t', time.time()-tbatch)
                print('Time per mask:\t\t', (time.time()-tbatch)/batch_size)
                savenum = i
                self.save_path = os.path.join(args0.save_path,'data'+str(savenum))
                os.makedirs(self.save_path,exist_ok=True)
                self.save_checkpoint(append=True, roi_only=True)
                zfile = open(os.path.join(self.save_path,'zcoeffs.txt'), 'a')
                np.savetxt(zfile, zlist, fmt='%d')
                zfile.close()
                self.init_metrics()
                zlist = []
                
                tbatch = time.time()
                
    
    def get_coeff_list(self,zmodes,coeffs):
        if not isinstance(zmodes,(list,np.ndarray)):
            zmodes, coeffs = [zmodes], [coeffs]
        cfs = np.zeros(49)
        for i, zmode in enumerate(zmodes):
            cfs[zmode] = coeffs[i]
        return cfs

    
    def get_best_coefficient(self, zmode, coeffs, method='poly', metric='spot', repeat=1, record_all_data=False, shuffle=True, save_roi=True):
        maxmets=[]
        spotmets=[]
        print('\ncoeff',end='')
        
        repcoeffs = []
        for i in range(repeat):
            if shuffle:
                np.random.shuffle(coeffs)
            repcoeffs.append(coeffs)
            for i,coeff in enumerate(coeffs):
                print('...'+str(coeff),end='')
                self.parent_masks.update_zernike_parents(self.get_coeff_list(zmode,coeff))

                self.interface.get_output_fields(self.parent_masks)
                self.update_metrics(update_type='initial',save_mask=False, save_roi=save_roi)
                maxmets.append(self.metrics['maxmet'][-1])
                spotmets.append(self.metrics['spot'][-1])

        repcoeffs = np.array(repcoeffs).flatten()
        srt_idxs = np.argsort(repcoeffs)
        for met, metlist in self.metrics.items():
            if not save_roi and 'roi' in met:
                    continue
            if 'mask' in met:
                continue
            sortvals = metlist[-len(repcoeffs):]
            metlist[-len(repcoeffs):] = [sortvals[zz] for zz in srt_idxs]
        print('\n')

        coeffs = repcoeffs[srt_idxs]
        maxmets = np.array(maxmets)[srt_idxs]
        spotmets = np.array(spotmets)[srt_idxs]
        if method == 'poly':
            max_coeff = get_polybest(coeffs, maxmets, np.argmax, plot=True, plotfolder=self.save_path, plotname='zmode_'+str(zmode)+'_max')
            spot_coeff = get_polybest(coeffs, spotmets, np.argmax, plot=True, plotfolder=self.save_path, plotname='zmode_'+str(zmode)+'_spot')
            if metric == 'max':
                best_coeff = int(max_coeff)
            elif metric == 'spot':
                best_coeff = int(spot_coeff)

        elif method == 'argmax':
            max_coeff = np.argmax(maxmets)
            spot_coeff = np.argmax(spotmets)
            if metric == 'max':
                best_coeff = coeffs[max_coeff]
            elif metric == 'spot':
                best_coeff = coeffs[spot_coeff]
        
        
##        if isinstance(best_coeff,np.ndarray):
##            best_coeff = best_coeff[-1]
##        return coeffs[best_coeff]
        print('best_coeff:', best_coeff, '\t best_coeff intensity:', maxmets[np.argmin(np.abs(coeffs - best_coeff))], '\t method:', method)
        print('maxmets max:', np.max(maxmets), '\t argmax coeff:', coeffs[np.argmax(maxmets)])

        if metric == 'max':
            xthresh_idxs = get_xthresh_idxs(coeffs, medfilt(maxmets, 3), thresh=0.7)
        elif metric == 'spot':
            xthresh_idxs = get_xthresh_idxs(coeffs, medfilt(spotmets, 3), thresh=0.7)

        next_coeffs = coeffs[xthresh_idxs]
        return best_coeff, next_coeffs

    
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
        file.close()
    
    def final_log(self):
        file = open(self.save_path+'/log.txt','a')
        file.write('Final Spot Metric: '+str(1/self.metrics['spot'][-1])+'\n')
        file.write('Final Spot Enhancement: '+str(self.metrics['spot'][0]/self.metrics['spot'][-1])+'\n')          
        file.write('Final Intensity Enhancement: '+str(self.metrics['maxint'][-1]/self.metrics['maxint'][0])+'\n\n')
        file.write('Optimization Time: '+str(self.get_time())+'\n')
        file.write('end time: '+str(datetime.datetime.now())+'\n')
        file.close()
        
        
    def save_checkpoint(self, append=False, roi_only=False):
        files = ['/spot_metric_vals_checkpoint.txt',
                 '/max_metric_vals_checkpoint.txt',
                 '/mean_intensity_vals_checkpoint.txt',
                 '/max_intensity_vals_checkpoint.txt',
                 '/roi.txt',
                 '/masks.txt']

        mode = 'w+'
        if append:
            mode = 'a'
        
        f = []

        if roi_only:
            f.append(open(self.save_path+'/roi.txt', mode))
            np.savetxt(f[0], self.metrics['roi'], fmt='%d')
        else:
            f = []
            for file in files:
                f.append(open(self.save_path+file, mode))
            np.savetxt(f[0], self.metrics['spot'], fmt='%.4e')
            np.savetxt(f[1], self.metrics['maxmet'], fmt='%10.3f')
            np.savetxt(f[2], self.metrics['mean'], fmt='%10.3f')
            np.savetxt(f[3], self.metrics['maxint'], fmt='%d')
            np.savetxt(f[4], self.metrics['roi'], fmt='%d')
            np.savetxt(f[5],self.metrics['masks'], fmt='%d')
            np.savetxt(self.save_path+'/bestmask.txt',self.parent_masks.get_slm_masks()[-1], fmt = '%d')
            if not isinstance(self.parent_masks.get_base_mask(),int):
                np.savetxt(self.save_path+'/base_mask.txt', np.squeeze(self.parent_masks.get_base_mask()), fmt = '%d')

        for x in f:
            x.close()

    def load_checkpoint(self, fdir=None, load_roi=True, load_masks=True):
        fdict = {'spot':'/spot_metric_vals_checkpoint.txt',
                 'maxmet':'/max_metric_vals_checkpoint.txt',
                 'mean':'/mean_intensity_vals_checkpoint.txt',
                 'maxint':'/max_intensity_vals_checkpoint.txt',
                 'roi':'/roi.txt',
                 'masks':'/masks.txt'}
        
        dtype = {'spot':np.float,
                 'maxmet': np.float,
                 'mean': np.float ,
                 'maxint': np.uint8,
                 'roi': np.uint8,
                 'masks': np.uint8}
        
        if fdir is None:
            fdir = self.save_path
        
        for met in fdict.keys():
            if (met == 'roi' and not load_roi) or (met == 'masks' and not load_masks):
                continue
            if os.path.isfile(self.save_path + fdict[met]):
                self.metrics[met] = list(np.loadtxt(self.save_path + fdict[met], dtype=dtype[met]))
            else:
                print('WARNING: Metric '+met+' not loaded.','\nfile not found:',self.save_path + fdict[met])
  
    
    def save_plots(self, fromfile=False, fdir=None):
        if fromfile is True:
            self.load_checkpoint()
        if fdir is None:
            fdir = self.save_path

        if len(self.metrics['maxmet']) > 1:
            plt.figure()
            plt.plot(self.metrics['maxmet'])
            plt.savefig(fdir+'/max_metric_plot')
            plt.close()

        if len(self.metrics['maxint']) > 1:
            plt.figure()
            plt.plot(self.metrics['maxint'])
            plt.savefig(fdir+'/max_intensity_plot')
            plt.close()

        if len(self.metrics['mean']) > 1:
            plt.figure()
            plt.plot(self.metrics['mean'])
            plt.savefig(fdir+'/mean_intensity_plot')
            plt.close()

        if len(self.metrics['spot']) > 1:
            plt.figure()
            plt.plot(self.metrics['spot'])
            plt.savefig(fdir+'/spot_metric_plot')
            plt.close()

        if len(self.metrics['roi']) > 1:
            dim=int(np.sqrt(len(self.metrics['roi'][0])))
            plt.figure()
            plt.imshow(np.array(self.metrics['roi'][-2]).reshape(dim,dim))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig(fdir+'/end_roi')
            plt.close()

            plt.figure()
            plt.imshow(np.array(self.metrics['roi'][1]).reshape(dim,dim))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig(fdir+'/begin_roi')
            plt.close()

        if len(self.metrics['roi']) > 0:
            dim=int(np.sqrt(len(self.metrics['roi'][0])))
            plt.figure()
            plt.imshow(np.array(self.metrics['roi'][-1]).reshape(dim,dim))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig(fdir+'/end_roi_averaged')
            plt.close()

            plt.figure()
            plt.imshow(np.array(self.metrics['roi'][0]).reshape(dim,dim))
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.savefig(fdir+'/begin_roi_averaged')
            plt.close()
        
        plt.figure(figsize=(12,8))
        bmask = np.array(self.parent_masks.get_slm_masks()[-1],dtype=np.uint8).reshape(self.args.slm_height,self.args.slm_width)
        plt.imshow(bmask, cmap='Greys_r')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.savefig(fdir+'/final_mask')
        plt.close()

    def save_zernike_mask(self, zmodes_file):
        coeffs = np.loadtxt(zmodes_file)
        print(zmodes_file)
        print(coeffs.shape, coeffs)
        zmask = self.parent_masks.create_zernike_mask(coeffs)
        np.savetxt(os.path.dirname(zmodes_file)+'/bestmask.txt',zmask, fmt='%d')

    
        
def get_mask_compare_list(directory,names=['bestmask'],write_to_file=True):
        maskfiles = []
        for root, dirs, files in os.walk(directory):
            for d in files:
                f = os.path.join(root,d)
                if 'compare_masks' in f or 'ignore' in f:
                    continue
                for name in names:
                    if name in f:
                        maskfiles.append(f)
                        print(f,'...added to compare list.')
        if write_to_file:
            with open(os.path.join(directory,'compare_list.txt'),'w+') as f:
                for m in maskfiles:
                    f.write(m+'\n')
        return maskfiles

def get_polybest(x,y,best_func=np.argmax, deg=4, plot=False, plotfolder=None, plotname=None):
    if best_func == np.argmin:
        y = 1/y
        best_func = np.argmax

    yfilt = medfilt(y, 5)
    xthresh_idxs = get_xthresh_idxs(x, yfilt)
    print('xthresh = ', xthresh_idxs , '\t diff', xthresh_idxs[0]-xthresh_idxs[1])
    pfit = np.polyfit(x[xthresh_idxs[0]:xthresh_idxs[1]],y[xthresh_idxs[0]:xthresh_idxs[1]],deg)
    p = np.poly1d(pfit)
    frange = np.arange(x[xthresh_idxs[0]],x[xthresh_idxs[1]],1)
    zbest = best_func(p(frange))
    print('zbest_idx:',zbest, '\t zbest_coeff:', int(frange[zbest]))
    print('len frange:', frange.shape)
    best_coeff = int(frange[zbest])

    if plot:
        minp = min(p(frange))
        maxp = max(p(frange))
        
        plt.figure()
        plt.scatter(x, y, s=20, label='data')
        plt.plot(x, yfilt, lw=1, c='g', ls='-', label='filtered_data')
        plt.plot(frange, p(frange), c='k', lw=2, ls = '-', label='fit')
        plt.plot([best_coeff,best_coeff],[minp*.95,p(best_coeff)], c='r', lw=2, ls='--', label='best')
        plt.ylim(minp*.95, maxp + (maxp-minp)*.2)
        plt.legend()
        plt.savefig(os.path.join(plotfolder,plotname+'.png'))
        plt.close()
    return int(frange[zbest])


def get_xthresh_idxs(x,y,thresh=0.7):
    '''Return the x indices where the function has decreased to the threshold from the max.'''
    ymax = np.max(y)
    ymaxidx = np.argmax(y)
    
    ylimval = ymax*thresh
    if ymaxidx > 0:
        idxleft = np.argmin(np.abs(y[:ymaxidx]-ylimval))
    else:
        idxleft = 0

    if ymaxidx < len(y)-1:
        idxright = np.argmin(np.abs(y[ymaxidx:]-ylimval))+ymaxidx
    else:
        idxright = ymaxidx
    
    return [idxleft,idxright]
    
