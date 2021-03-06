import numpy as np
from Zernike import Zernike
import sys, os, argparse, copy, math

class Population:
    def __init__(self, args, base_mask=0, uniform=False):
        self.args = args
        self.num_masks = args.num_masks
        self.num_childs = args.num_childs
        self.slm_height = args.slm_height
        self.slm_width = args.slm_width
        self.segment_rows = int(args.slm_height/args.segment_height)
        self.segment_cols = int(args.slm_width/args.segment_width)
        self.segment_height = args.segment_height
        self.segment_width = args.segment_width
        self.num_phase_vals = args.num_phase_vals
        self.fitness_func = args.fitness_func
        self.uniform = uniform
        self.uniform_childs = args.add_uniform_childs
        self.num_uniform = 1
        self.uniform_parent_prob = args.uniform_parent_prob

        # new args
        self.masktype = args.masktype
        self.zernike_modes = args.zernike_modes

        self.mutate_initial_rate = args.mutate_initial_rate
        self.mutate_final_rate = args.mutate_final_rate
        self.mutate_decay_factor = args.mutate_decay_factor
        
        self.zernike_coeffs = args.zernike_coeffs # list of coefficients for modes 3 up to 15
        self.grating_step = args.grating_step

        pstep = 1/sum(np.arange(self.num_masks+1)) # Increment for generating probability distribution.
        self.parent_probability_dist = np.arange(pstep,(self.num_masks+1)*pstep,pstep) # Probability distribution for parent selection.
        if self.masktype == 'rect':
            self.phase_vals = np.arange(0,args.num_phase_vals,1,dtype=np.float) # Distribution of phase values for SLM
        if self.masktype == 'zernike':
            self.phase_vals = np.arange(-args.num_phase_vals,args.num_phase_vals,1)# Dist of zmode coeff values

        self.base_mask = base_mask
        self.zernike_mask = 0
        self.grating_mask = 0
        self.zbasis = None
        self.init_masks()
        self.init_zernike_mask()
        self.init_grating_mask()
        self.fitness_vals = []
        self.output_fields = []
        
    def get_masks(self):
        """Return mask list."""
        return self.masks

    def get_base_mask(self):
        return self.base_mask
    
    def get_output_fields(self):
        return self.output_fields
    
    def get_fitness_vals(self):
        return self.fitness_vals
    
    def add_mask(self, mask, fitness=None):
        self.masks.append(mask)
    
    def get_replace_idx(self, fitness):
        for j, val in enumerate(self.fitness_vals):
            if val<fitness:
                return j
        return None

    def save_masks(self, directory, name='masks_population'):
        masks = np.array(self.masks).reshape(-1, self.segment_rows*self.segment_cols)
        np.savetxt(os.path.join(directory, name+'.txt'), masks, fmt = '%d')

    def load_masks(self, directory, name='masks_population'):
        files = next(os.walk(directory))[2]
        for f in files:
            if name in f:
                path = os.path.join(directory,f)
        
        print('Loading masks from file...\n', path)
        masks = np.loadtxt(path).reshape(-1, self.segment_rows, self.segment_cols).astype(np.float)
        self.masks = list(masks)
        print('Success!')
            
    
    def replace_parents(self,children):
        cmasks = children.get_masks()
        cfields = children.get_output_fields()
        
        for i, cval in enumerate(children.get_fitness_vals()):
            idx = self.get_replace_idx(cval)
            if not (idx is None):
                self.masks[idx]=cmasks[i]
                self.fitness_vals[idx]=cval
                self.output_fields[idx]=cfields[i]
        self.ranksort()
                
    
    def update_masks(self,new_masks):
        self.masks = new_masks
        
    def update_base_mask(self,new_mask):
        self.base_mask = new_mask
                
    def update_output_fields(self,output_fields):
        self.output_fields = output_fields
    
    def get_slm_masks(self):
        """Return masks to be loaded onto slm."""
        if self.masktype == 'rect':
            slm_masks = np.array([self.create_full_mask(mask) for mask in self.masks])
        else:
            slm_masks = np.array(self.masks)
        slm_masks += self.base_mask + self.zernike_mask + self.grating_mask
        return np.mod(np.round(slm_masks,0),256).astype(dtype=np.uint8)
        
    def create_mask(self,uniform_bool=None, masktype=None):
        if uniform_bool is None:
            uniform_bool = self.uniform

        if masktype == None:
            masktype = self.masktype
        
        if masktype == 'rect':
            newmask = np.zeros((self.segment_rows, self.segment_cols),dtype=np.float)
        if masktype == 'zernike':
            newmask = np.zeros(len(self.zernike_modes))
        
        if not uniform_bool:
            for i in range(int(newmask.size*self.mutate_initial_rate)):
                newmask[tuple([np.random.randint(0, x) for x in newmask.shape])] = np.random.choice(self.phase_vals)
        return newmask
    
    def create_full_mask(self,mask, masktype=None):
        if masktype is None:
            masktype = self.masktype
            
        if masktype == 'rect':
            if np.shape(mask)[0] == self.slm_height:
                return mask
            else:
                segment = np.ones((self.segment_height, self.segment_width),dtype=np.float)
                return np.kron(mask,segment)
            
        if masktype == 'zernike':
            return self.create_zernike_mask(zcoeffs=mask,zmodes=self.zernike_modes)
            
############################################### Begin Zernike ########################################
    
    def init_zernike_mask(self):
        '''Initialize zernike base mask.'''
        self.zernike = Zernike(self.slm_width, self.slm_height)
        self.zernike_mask = self.create_zernike_mask()

    def init_zbasis(self):
        '''Initialize set of zernike basis functions for fast calculation of new zernike masks.'''
        print('Initializing zernike basis functions...')
        zmodes = np.arange(0,49)
        self.zbasis = np.zeros((len(zmodes),self.slm_height*self.slm_width), dtype=np.float)
        for i, z in enumerate(zmodes):
            zcoeffs = (zmodes*0)
            zcoeffs[i] = 1
            zfunc = self.create_zernike_mask(zcoeffs, dtype=np.float, zbasis=False)
            if np.max(np.abs(zfunc)) > 0:
                self.zbasis[i] = zfunc.flatten()

        print('zbasis shape:',self.zbasis.shape)
    
    def change_parent_zcoeff(self,newcoeff):
        '''Rescale zernike base mask if only single nonzero coefficient.'''
        if self.single_zcoeff == True:
            self.masks = [(mask.astype(np.float)/self.single_zcoeff_val*newcoeff).astype(np.float) for mask in self.masks]
            self.single_zcoeff_val = newcoeff
        else:
            print('Warning: zernike mask has more than one coefficient. Cannot rescale!')
        
    def update_zernike_parents(self,zcoeffs=None, zbasis=True):
        ''' '''
        if zcoeffs is None:
            zcoeffs = self.zernike_coeffs
            
        zcoeffs=np.array(zcoeffs,dtype=np.int)
        if zcoeffs.ndim == 1:
            zcoeffs = zcoeffs.reshape(1,-1)
            
        self.masks = self.create_zernike_mask(zcoeffs, zbasis=zbasis)
    
    def create_zernike_mask(self, zcoeffs=None, zmodes=None, dtype=np.float, zbasis=False):
        if zcoeffs is None:
            zcoeffs = self.zernike_coeffs
        if zmodes is None:
            zmodes = np.arange(len(zcoeffs))
            
        zcoeffs = np.array(zcoeffs)
        if max(zcoeffs.shape) == 1:
            zcoeffs = np.zeros((1,49))
            
        zcoeffs = zcoeffs.reshape(-1,zcoeffs.shape[-1])
            
        if zbasis:
            if self.zbasis is None:
                self.init_zbasis()
            newmask = np.matmul(zcoeffs, self.zbasis)
        else:
            newmask = np.zeros((np.shape(zcoeffs)[0],self.slm_height,self.slm_width))
            for m in range(np.shape(zcoeffs)[0]):
                for i,coefficient in enumerate(zcoeffs[m]):
                    if coefficient != 0 and zmodes[i]>= 1 and zmodes[i]<=48:
                        func = getattr(self.zernike,'z'+str(zmodes[i]))
                        zmask = np.fromfunction(func,(self.slm_height, self.slm_width),dtype=np.float)
                        zmask *= coefficient
                        newmask[m] += zmask

        return newmask.reshape(-1,self.slm_height, self.slm_width)

####################################### End Zernike #######################################################################

####################################### Begin Grating #######################################################################
    
    def init_grating_mask(self):
        self.grating_mask = self.create_grating_mask()
    
    def update_grating_mask(self,step=None,u=False):
        self.grating_mask = self.create_grating_mask(step,u)
        
    
    def create_grating_mask(self,step=None,u=False):
        newmask = self.create_full_mask(self.create_mask(True))
        if step is None:
            step = self.grating_step
        if step==0:
            return newmask
        pattern = np.arange(0,int(self.slm_width),1,dtype=np.float)*step
        newmask += np.kron(pattern,np.ones((int(self.slm_height),1),dtype=np.float))
        return newmask
    
####################################### End Grating #######################################################################

    def init_masks(self):
        self.masks=[]
        for i in range(self.num_masks):
            if self.uniform_childs and (i>=(self.num_masks-2) or i<2):
                self.masks.append(self.create_mask(True))
            else:
                self.masks.append(self.create_mask())
                

    def update_fitness_vals(self,scale=0):
        if scale != 0:
            uniform_intensity = np.mean([self.output_fields[:self.num_uniform],self.output_fields[-self.num_uniform:]]) # mean intensity of uniform masks' output fields
            outfields = (self.output_fields.astype(np.float)*scale/uniform_intensity).astype(np.int)
            self.fitness_vals = [self.fitness(field) for field in outfields]
        else:
            self.fitness_vals = [self.fitness(field) for field in self.output_fields]

            
            
    def ranksort(self, scale=0):
        """Sort masks by fitness value"""
        self.update_fitness_vals()
        if len(self.masks)>1:
            ranks = np.argsort(self.fitness_vals)
            self.fitness_vals = np.array(self.fitness_vals)[ranks].tolist()
            self.masks = np.array(self.masks)[ranks].tolist()
            self.output_fields = np.array(self.output_fields,dtype=np.int)[ranks].tolist()
        
    
    def make_children(self,add_uniform=False):
        child_args = copy.copy(self.args)
        child_args.num_masks = self.num_childs
        child_args.zernike_coeffs = self.zernike_coeffs
        children = Population(child_args,self.base_mask)
        children.zbasis = self.zbasis
        new_masks = [self.breed() for i in range(self.num_childs)]
        if add_uniform:
            for i in range(self.num_uniform):
                new_masks.append(self.create_mask(True))
        children.update_masks(new_masks)
        return children

    def update_num_mutations(self,gen,numgens):
        if self.masktype == 'rect':
            num_segments = int(self.segment_rows*self.segment_cols)
        if self.masktype == 'zernike':
            num_segments = int(len(self.zernike_modes))
        self.num_mutations = int(round(num_segments * ((self.mutate_initial_rate - self.mutate_final_rate)
                                                    * np.exp(-gen / self.mutate_decay_factor)
                                                    + self.mutate_final_rate)))
        self.num_mutations = max(1,self.num_mutations)
        
    def breed(self):
        """Breed two "parent" masks and return new mutated "child" input mask array."""
        pidx = np.random.choice(len(self.masks),size=2,replace=False,p=self.parent_probability_dist)
        parents = [np.array(self.masks[p],dtype=np.float) for p in pidx]
        if self.uniform_childs:
            if np.random.choice([True,False],p=[self.uniform_parent_prob,1 - self.uniform_parent_prob]):
                parents[0]=self.create_mask(True)
        shape = parents[0].shape
        rand_matrix = np.random.choice([True,False],size=shape)
        child = parents[0]*rand_matrix+parents[1]*np.invert(rand_matrix)
        for i in range(self.num_mutations):
            idx = tuple([np.random.randint(0,x) for x in shape])
            child[idx] = np.random.choice(self.phase_vals)
        return child

    def fitness(self, output_field,func=None):
        """Return the mean of output_field.

        Note: Adjust fitness function to suit your optimization process.
        """
        if func is None:
            func = self.fitness_func
        if func == 'localmax':
            output_field = np.asarray(output_field)
            dim = int(np.sqrt(output_field.shape[0]))
            output_field = output_field.reshape(dim,dim)
            d=2
            midx = np.argmax(output_field[d:dim-d,d:dim-d])
            midx = np.array([midx%(dim-2*d),math.floor(midx/(dim-2*d))]) + d
           
            row = (midx[0]-d,midx[0]+d+1)
            col = (midx[1]-d,midx[1]+d+1)
            cen = output_field[max(row[0],0):min(row[1],dim),max(col[0],0):min(col[1],dim)]

            wroi = np.zeros(cen.shape) + .1/16
            wroi[1:-1,1:-1] = 0.2/9
            wroi[d,d] = 0.7

            wcen = np.multiply(wroi,cen)
            return np.sum(np.multiply(wroi,cen))
            
        if func == 'max':
            return weighted_max_metric(output_field)

        if func == 'spot':
            return spot_metric(output_field)
##            return weighted_log_metric(max_metric(output_field),spot_metric(output_field))


        if func == 'mean':
            return np.mean(output_field)

        print('Invalid Fitness Function...')
    
def weighted_log_metric(maxmet,spotmet):
    maxmet = np.maximum(np.array(maxmet),1)  # set max(log(maxmet)) = 0
    spotmet = np.array(spotmet)
    return np.log(maxmet)*spotmet

def spot_metric(output_field):
    output_field = np.array(output_field, dtype=np.float)
    a = 1000
    return (spot(output_field + a) - spot(np.ones(output_field.size))) * output_field.size**2

def spot(output_field):
    return np.sum(np.square(output_field))/np.sum(output_field)**2

def spot_metric_denoised(output_field):
    output_field = np.array(output_field)
    dnoise = 20
    output_field = np.maximum(output_field - dnoise, 0)
    output_field[output_field > 0] += dnoise
    return np.sum(np.square(output_field + 1)) / np.sum(output_field + 1) ** 2

def weighted_max_metric(output_field):
    output_field = np.asarray(output_field)
    output_field = output_field[np.argsort(output_field)]
    return np.mean(
        np.mean(output_field[-22:-10])*.05
        + np.mean(output_field[-10:-5])*.1
        + np.mean(output_field[-5:-1])*.2
        + output_field[-1]*.65)    

