import numpy as np
import win32pipe as wp
import win32file as wf
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import time, datetime, sys, os, argparse, copy, math

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
        self.num_uniform = 2

        self.mutate_initial_rate = args.mutate_initial_rate
        self.mutate_final_rate = args.mutate_final_rate
        self.mutate_decay_factor = args.mutate_decay_factor
        
        self.zernike_coeffs = args.zernike_coeffs # list of coefficients for modes 3 up to 15
        self.grating_step = args.grating_step

        pstep = 1/sum(np.arange(self.num_masks+1)) # Increment for generating probability distribution.
        self.parent_probability_dist = np.arange(pstep,(self.num_masks+1)*pstep,pstep) # Probability distribution for parent selection.
        self.phase_vals = np.arange(0,args.num_phase_vals,1,dtype=np.uint8) # Distribution of phase values for SLM
        
        self.base_mask = base_mask
        self.zernike_mask = 0
        self.grating_mask = 0
        
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
        self.output_fields = np.array(output_fields,dtype=np.int)
    
    def get_slm_masks(self):
        """Return masks to be loaded onto slm."""
##        print(type(self.create_full_mask(self.masks[0])),np.shape(self.base_mask), type(self.zernike_mask), type(self.grating_mask))
        slm_masks = [self.create_full_mask(mask) + self.base_mask + self.zernike_mask + self.grating_mask for mask in self.masks]
        return slm_masks
        
    def create_mask(self,uniform_bool=None):
        if uniform_bool is None:
            uniform_bool = self.uniform
        newmask = np.zeros((self.segment_rows, self.segment_cols),dtype=np.uint8)
        if uniform_bool == False:
            for i in range(int(self.segment_rows*self.segment_cols*self.mutate_initial_rate)):
                newmask[np.random.randint(0, self.segment_rows), np.random.randint(0,self.segment_cols)] = np.random.choice(self.phase_vals)
        return newmask
    
    def create_full_mask(self,mask):
        if np.shape(mask)[0] == self.slm_height:
            return mask
        else:
            segment = np.ones((self.segment_height, self.segment_width),dtype=np.uint8)
            return np.kron(mask,segment)
            
############################################### Begin Zernike ########################################
    
    def init_zernike_mask(self):
        self.zernike = Zernike(self)
        self.zernike_mask = self.create_zernike_mask()
##        plt.imshow(self.zernike_mask)
##        plt.colorbar()
##        plt.show()
    
    def change_parent_zcoeff(self,newcoeff):
        if self.single_zcoeff == True:
            self.masks = [(mask/self.single_zcoeff_val*newcoeff).astype(np.uint8) for mask in self.masks]
            self.single_zcoeff_val=newcoeff
        else:
            print('Warning: zernike mask has more than one coefficient. Cannot rescale!')
        
    def update_zernike_parent(self,zcoeffs=None):
        if zcoeffs is None:
            zcoeffs = self.zernike_coeffs
        zcoeffs=np.array(zcoeffs,dtype=np.int)
        if max(np.shape(np.nonzero(zcoeffs)))==1:
            self.single_zcoeff = True
            self.single_zcoeff_val = zcoeffs[np.nonzero(zcoeffs)]
        else:
            self.single_zcoeff = False
        self.masks = [self.create_zernike_mask(zcoeffs)]
    
    def create_zernike_mask(self,zcoeffs=None):
        if zcoeffs is None:
            zcoeffs = self.zernike_coeffs
        newmask = self.create_full_mask(self.create_mask(True))
        for i,coefficient in enumerate(zcoeffs):
            if coefficient != 0:
                #print('mode '+str(i),coefficient)
                num = i+3
                func = getattr(self.zernike,'z'+str(num))
                zmask = np.fromfunction(func,(self.slm_height, self.slm_width))
                zmask *= 4*coefficient/np.max(np.abs(zmask))
                newmask += zmask.astype(np.uint8)
        return np.array(newmask,dtype=np.uint8)

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
        pattern = np.arange(0,int(self.slm_width),1,dtype=np.uint8)*step
        newmask += np.kron(pattern,np.ones((int(self.slm_height),1),dtype=np.uint8))
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
##            self.output_fields = (self.output_fields.astype(np.float)*scale/uniform_intensity).astype(np.int)
            outfields = (self.output_fields.astype(np.float)*scale/uniform_intensity).astype(np.int)
            self.fitness_vals = [self.fitness(field) for field in outfields]
##            print('\nscale1',uniform_intensity)
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
        new_masks = [self.breed() for i in range(self.num_childs)]
        if add_uniform:
            for i in range(self.num_uniform):
                new_masks.append(self.create_mask(True))
                new_masks.insert(0,self.create_mask(True))
        children.update_masks(new_masks)
        return children

    def update_num_mutations(self,gen,numgens):
        num_segments = int(self.segment_rows*self.segment_cols)
        self.num_mutations = int(round(num_segments * ((self.mutate_initial_rate - self.mutate_final_rate)
                                                    * np.exp(-gen / self.mutate_decay_factor)
                                                    + self.mutate_final_rate)))
        self.num_mutations = max(1,self.num_mutations)
        
    def breed(self):
        """Breed two "parent" masks and return new mutated "child" input mask array."""
        pidx = np.random.choice(len(self.masks),size=2,replace=False,p=self.parent_probability_dist)
        parents = [np.array(self.masks[pidx[0]],dtype=np.uint8),np.array(self.masks[pidx[1]],dtype=np.uint8)]
        if self.uniform_childs:
            uprob = 0.1
            if np.random.choice([True,False],p=[uprob,1-uprob]):
                parents[0]=self.create_mask(True)
        shape = parents[0].shape
        rand_matrix = np.random.choice([True,False],size=shape).reshape(shape)
        child = parents[0]*rand_matrix+parents[1]*np.invert(rand_matrix)
        for i in range(self.num_mutations):
            child[np.random.randint(0,shape[0]),np.random.randint(0,shape[1])] = np.random.choice(self.phase_vals)
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
            output_field = np.asarray(output_field)
            midxs = np.argsort(output_field)
            output_field = output_field[np.argsort(output_field)]
            return np.mean(
                np.mean(output_field[-22:-10])*.05
                + np.mean(output_field[-10:-5])*.1
                + np.mean(output_field[-5:-1])*.2
                + output_field[-1]*.65)
            # Trial max with new weights
##            return np.mean(
##                    np.mean(output_field[-22:-10])*.7
##                    + np.mean(output_field[-10:-5])*.8
##                    + np.mean(output_field[-5:-1])*.9
##                    + output_field[-1])            

        if func == 'spot':
            if np.sum(output_field)==0:
                return 0
            return np.sum(np.square(output_field))/np.sum(output_field)**2

        if func == 'mean':
            return np.mean(output_field)

        print('Invalid Fitness Function...')
    

                  

        
        
class Zernike:
    def __init__(self, population):
        self.x0 = int(population.segment_cols*population.segment_width/2)
        self.y0 = int(population.segment_rows*population.segment_height/2)
        self.scale = 1/self.y0

    def z3(self,y,x):
        return -1 + 2*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)

    def z4(self,y,x):
        return (((x-self.x0)*self.scale)**2-((y-self.y0)*self.scale)**2)

    def z5(self,y,x):
        return 2*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)

    def z6(self,y,x):
        return -2*((x-self.x0)*self.scale) + 3*((x-self.x0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)

    def z7(self,y,x):
        return -2*((y-self.y0)*self.scale) + 3*((y-self.y0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)

    def z8(self,y,x):
        return 1 - 6*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2) + 6*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)**2

    def z9(self,y,x):
        return ((x-self.x0)*self.scale)**3 - 3*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**2

    def z10(self,y,x):
        return -((y-self.y0)*self.scale)**2 + 3*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)

    def z11(self,y,x):
        return -3*((x-self.x0)*self.scale)**2 + 3*((y-self.y0)*self.scale)**2 + 4*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2) - 4*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)

    def z12(self,y,x):
        return -6*((x-self.x0)*self.scale)*((y-self.y0)*self.scale) + 8*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)

    def z13(self,y,x):
        return 3*((x-self.x0)*self.scale) - 12*((x-self.x0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2) + 10*((x-self.x0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)**2

    def z14(self,y,x):
        return 3*((y-self.y0)*self.scale) - 12*((y-self.y0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2) + 10*((y-self.y0)*self.scale)*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)**2

    def z15(self,y,x):
        return -1 + 12*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)- 30*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)**2 + 20*(((x-self.x0)*self.scale)**2+((y-self.y0)*self.scale)**2)**3

    def z16(self,y,x):
        return ((x-self.x0)*self.scale)**4 - 6*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2 + ((y-self.y0)*self.scale)**4

    def z17(self,y,x):
        return 4 * ((x-self.x0)*self.scale)**3 * ((y-self.y0)*self.scale) - 4*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**3

    def z18(self,y,x):
        return -4*((x-self.x0)*self.scale)**3 + 12*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**2 + 5*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 15*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

    def z19(self,y,x):
        return -12 *((x-self.x0)*self.scale)**2 *((y-self.y0)*self.scale) + 4 *((y-self.y0)*self.scale)**3 + 15 *((x-self.x0)*self.scale)**2 *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 5 *((y-self.y0)*self.scale)**3 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

    def z20(self,y,x):
        return  6 *((x-self.x0)*self.scale)**2 - 6 *((y-self.y0)*self.scale)**2 - 20 *((x-self.x0)*self.scale)**2 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 20 *((y-self.y0)*self.scale)**2 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 15 *((x-self.x0)*self.scale)**2 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 15 *((y-self.y0)*self.scale)**2 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

    def z21(self,y,x):
        return  12 *((x-self.x0)*self.scale) *((y-self.y0)*self.scale) - 40 *((x-self.x0)*self.scale) *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 30 *((x-self.x0)*self.scale) *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2


    def z22(self,y,x):
        return -4 *((x-self.x0)*self.scale) + 30 *((x-self.x0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 60 *((x-self.x0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 35 *((x-self.x0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

    def z23(self,y,x):
        return -4 *((y-self.y0)*self.scale) + 30 *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 60 *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 35 *((y-self.y0)*self.scale) *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3


    def z24(self,y,x):
        return 1 - 20 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 90 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 140* (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 70 *(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

    def z25(self,y,x):
        return  ((x-self.x0)*self.scale)**5 - 10 *((x-self.x0)*self.scale)**3 *((y-self.y0)*self.scale)**2 + 5*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**4


    def z26(self,y,x):
        return 5 *((x-self.x0)*self.scale)**4 *((y-self.y0)*self.scale) - 10 *((x-self.x0)*self.scale)**2 *((y-self.y0)*self.scale)**3 + ((y-self.y0)*self.scale)**5
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
