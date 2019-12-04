#!/usr/bin/env python


class Zernike:
	def __init__(self, width, height):
		self.x0 = width/2
		self.y0 = height/2
		self.scale = 1/(self.y0**2 + self.x0**2)**0.5
		self.x0 -= 0.5
		self.y0 -= 0.5


	def z0(self,y,x):
		return 1

	def z1(self,y,x):
		return ((x-self.x0)*self.scale)

	def z2(self,y,x):
		return ((y-self.y0)*self.scale)

	def z3(self,y,x):
		return -1 + 2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z4(self,y,x):
		return ((x-self.x0)*self.scale)**2 - ((y-self.y0)*self.scale)**2

	def z5(self,y,x):
		return 2*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)

	def z6(self,y,x):
		return -2*((x-self.x0)*self.scale) + 3*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z7(self,y,x):
		return -2*((y-self.y0)*self.scale) + 3*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z8(self,y,x):
		return 1 - 6*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 6*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z9(self,y,x):
		return ((x-self.x0)*self.scale)**3 - 3*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**2

	def z10(self,y,x):
		return 3*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) - ((y-self.y0)*self.scale)**3

	def z11(self,y,x):
		return -3*((x-self.x0)*self.scale)**2 + 3*((y-self.y0)*self.scale)**2 + 4*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 4*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z12(self,y,x):
		return -6*((x-self.x0)*self.scale)*((y-self.y0)*self.scale) + 8*((x-self.x0)*self.scale)*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z13(self,y,x):
		return 3*((x-self.x0)*self.scale) - 12*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 10*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z14(self,y,x):
		return 3*((y-self.y0)*self.scale) - 12*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 10*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z15(self,y,x):
		return -1 + 12*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 30*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 20*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z16(self,y,x):
		return ((x-self.x0)*self.scale)**4 - 6*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2 + ((y-self.y0)*self.scale)**4

	def z17(self,y,x):
		return 4*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) - 4*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**3

	def z18(self,y,x):
		return -4*((x-self.x0)*self.scale)**3 + 12*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2 + 5*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 15*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z19(self,y,x):
		return -12*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) + 4*((y-self.y0)*self.scale)**3 + 15*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 5*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z20(self,y,x):
		return 6*((x-self.x0)*self.scale)**2 - 6*((y-self.y0)*self.scale)**2 - 20*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 20*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 15*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 15*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z21(self,y,x):
		return 12*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) - 40*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 30*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z22(self,y,x):
		return -4*((x-self.x0)*self.scale) + 30*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 60*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 35*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z23(self,y,x):
		return -4*((y-self.y0)*self.scale) + 30*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 60*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 35*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z24(self,y,x):
		return 1 - 20*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 90*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 140*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 70*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

	def z25(self,y,x):
		return ((x-self.x0)*self.scale)**5 - 10*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale)**2 + 5*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**4

	def z26(self,y,x):
		return 5*((x-self.x0)*self.scale)**4*((y-self.y0)*self.scale) - 10*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**3 + ((y-self.y0)*self.scale)**5

	def z27(self,y,x):
		return -5*((x-self.x0)*self.scale)**4 + 30*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2 - 5*((y-self.y0)*self.scale)**4 + 6*((x-self.x0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 36*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 6*((y-self.y0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z28(self,y,x):
		return -20*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) + 20*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**3 + 24*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 24*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z29(self,y,x):
		return 10*((x-self.x0)*self.scale)**3 - 30*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2 - 30*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 90*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 21*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 63*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z30(self,y,x):
		return 30*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) - 10*((y-self.y0)*self.scale)**3 - 90*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 30*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 63*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 21*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z31(self,y,x):
		return -10*((x-self.x0)*self.scale)**2 + 10*((y-self.y0)*self.scale)**2 + 60*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 60*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 105*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 105*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 56*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 56*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z32(self,y,x):
		return -20*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) + 120*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 210*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 112*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z33(self,y,x):
		return 5*((x-self.x0)*self.scale) - 60*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 210*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 280*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 126*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

	def z34(self,y,x):
		return 5*((y-self.y0)*self.scale) - 60*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 210*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 280*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 126*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

	def z35(self,y,x):
		return -1 + 30*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 210*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 560*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 630*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4 + 252*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**5

	def z36(self,y,x):
		return ((x-self.x0)*self.scale)**6 - 15*((x-self.x0)*self.scale)**4*((y-self.y0)*self.scale)**2 + 15*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**4 - ((y-self.y0)*self.scale)**6

	def z37(self,y,x):
		return 6*((x-self.x0)*self.scale)**5*((y-self.y0)*self.scale) - 20*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale)**3 + 6*((x-self.x0)*self.scale)*((y-self.y0)*self.scale)**5

	def z38(self,y,x):
		return -6*((x-self.x0)*self.scale)**5 + 60*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale)**2 - 30*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**4 + 7*((x-self.x0)*self.scale)**5*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 70*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 35*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z39(self,y,x):
		return -30*((x-self.x0)*self.scale)**4*((y-self.y0)*self.scale) + 60*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**3 - 6*((y-self.y0)*self.scale)**5 + 35*((x-self.x0)*self.scale)**4*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 70*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 7*((y-self.y0)*self.scale)**5*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)

	def z40(self,y,x):
		return 15*((x-self.x0)*self.scale)**4 - 90*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2 + 15*((y-self.y0)*self.scale)**4 - 42*((x-self.x0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 252*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 42*((y-self.y0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 28*((x-self.x0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 168*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 28*((y-self.y0)*self.scale)**4*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z41(self,y,x):
		return 60*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) - 60*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**3 - 168*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 168*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 112*((x-self.x0)*self.scale)**3*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 112*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2

	def z42(self,y,x):
		return -20*((x-self.x0)*self.scale)**3 + 60*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2 + 105*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 315*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 168*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 504*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 84*((x-self.x0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 252*((x-self.x0)*self.scale) ((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z43(self,y,x):
		return -60*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) + 20*((y-self.y0)*self.scale)**3 + 315*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 105*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 504*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 168*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 252*((x-self.x0)*self.scale)**2*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 84*((y-self.y0)*self.scale)**3*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3

	def z44(self,y,x):
		return 15*((x-self.x0)*self.scale)**2 - 15*((y-self.y0)*self.scale)**2 - 140*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 140*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 420*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 420*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 504*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 504*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 210*((x-self.x0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4 - 210*((y-self.y0)*self.scale)**2*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

	def z45(self,y,x):
		return 30*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) - 280*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 840*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 1008*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 420*((x-self.x0)*self.scale) ((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4

	def z46(self,y,x):
		return -6*((x-self.x0)*self.scale) + 105*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 560*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 1260*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 1260*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4 + 462*((x-self.x0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**5

	def z47(self,y,x):
		return -6*((y-self.y0)*self.scale) + 105*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) - 560*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 + 1260*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 - 1260*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4 + 462*((y-self.y0)*self.scale) * (((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**5

	def z48(self,y,x):
		return 1 - 42*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2) + 420*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**2 - 1680*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**3 + 3150*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**4 - 2772*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**5 + 924*(((x-self.x0)*self.scale)**2 + ((y-self.y0)*self.scale)**2)**6

