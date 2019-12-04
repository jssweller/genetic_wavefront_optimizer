#!/usr/bin/env python

from math import sqrt

class Zernike:
	def __init__(self, width, height):
		self.x0 = width/2
		self.y0 = height/2)
		self.scale = 1/sqrt(self.y0**2 + self.x0**2)
		self.x0 -= 0.5
		self.y0 -= 0.5


	def z0(self,y,x):
		return 1

	def z1(self,y,x):
		return (self.x-self.x0)*self.scale

	def z2(self,y,x):
		return (self.y-self.y0)*self.scale

	def z3(self,y,x):
		return -1 + 2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z4(self,y,x):
		return (self.x-self.x0)*self.scale**2 - (self.y-self.y0)*self.scale**2

	def z5(self,y,x):
		return 2*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale

	def z6(self,y,x):
		return -2*(self.x-self.x0)*self.scale + 3*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z7(self,y,x):
		return -2*(self.y-self.y0)*self.scale + 3*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z8(self,y,x):
		return 1 - 6*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 6*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z9(self,y,x):
		return (self.x-self.x0)*self.scale**3 - 3*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale**2

	def z10(self,y,x):
		return 3*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale - (self.y-self.y0)*self.scale**3

	def z11(self,y,x):
		return -3*(self.x-self.x0)*self.scale**2 + 3*(self.y-self.y0)*self.scale**2 + 4*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 4*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z12(self,y,x):
		return -6*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale + 8*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z13(self,y,x):
		return 3*(self.x-self.x0)*self.scale - 12*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 10*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z14(self,y,x):
		return 3*(self.y-self.y0)*self.scale - 12*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 10*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z15(self,y,x):
		return -1 + 12*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 30*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 20*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z16(self,y,x):
		return (self.x-self.x0)*self.scale**4 - 6*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2 + (self.y-self.y0)*self.scale**4

	def z17(self,y,x):
		return 4*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale - 4*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale**3

	def z18(self,y,x):
		return -4*(self.x-self.x0)*self.scale**3 + 12*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2 + 5*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 15*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z19(self,y,x):
		return -12*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale + 4*(self.y-self.y0)*self.scale**3 + 15*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 5*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z20(self,y,x):
		return 6*(self.x-self.x0)*self.scale**2 - 6*(self.y-self.y0)*self.scale**2 - 20*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 20*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 15*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 15*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z21(self,y,x):
		return 12*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale - 40*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 30*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z22(self,y,x):
		return -4*(self.x-self.x0)*self.scale + 30*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 60*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 35*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z23(self,y,x):
		return -4*(self.y-self.y0)*self.scale + 30*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 60*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 35*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z24(self,y,x):
		return 1 - 20*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 90*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 140*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 70*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4

	def z25(self,y,x):
		return (self.x-self.x0)*self.scale**5 - 10*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale**2 + 5*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale**4

	def z26(self,y,x):
		return 5*(self.x-self.x0)*self.scale**4*(self.y-self.y0)*self.scale - 10*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**3 + (self.y-self.y0)*self.scale**5

	def z27(self,y,x):
		return -5*(self.x-self.x0)*self.scale**4 + 30*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2 - 5*(self.y-self.y0)*self.scale**4 + 6*(self.x-self.x0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 36*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 6*(self.y-self.y0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z28(self,y,x):
		return -20*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale + 20*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**3 + 24*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 24*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z29(self,y,x):
		return 10*(self.x-self.x0)*self.scale**3 - 30*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2 - 30*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 90*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 21*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 63*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z30(self,y,x):
		return 30*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale - 10*(self.y-self.y0)*self.scale**3 - 90*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 30*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 63*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 21*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z31(self,y,x):
		return -10*(self.x-self.x0)*self.scale**2 + 10*(self.y-self.y0)*self.scale**2 + 60*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 60*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 105*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 105*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 56*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 56*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z32(self,y,x):
		return -20*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale + 120*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 210*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 112*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z33(self,y,x):
		return 5*(self.x-self.x0)*self.scale - 60*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 210*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 280*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 126*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4

	def z34(self,y,x):
		return 5*(self.y-self.y0)*self.scale - 60*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 210*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 280*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 126*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4

	def z35(self,y,x):
		return -1 + 30*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 210*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 560*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 630*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4 + 252*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**5

	def z36(self,y,x):
		return (self.x-self.x0)*self.scale**6 - 15*(self.x-self.x0)*self.scale**4*(self.y-self.y0)*self.scale**2 + 15*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**4 - (self.y-self.y0)*self.scale**6

	def z37(self,y,x):
		return 6*(self.x-self.x0)*self.scale**5*(self.y-self.y0)*self.scale - 20*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale**3 + 6*(self.x-self.x0)*self.scale*(self.y-self.y0)*self.scale**5

	def z38(self,y,x):
		return -6*(self.x-self.x0)*self.scale**5 + 60*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale**2 - 30*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**4 + 7*(self.x-self.x0)*self.scale**5*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 70*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 35*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z39(self,y,x):
		return -30*(self.x-self.x0)*self.scale**4*(self.y-self.y0)*self.scale + 60*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**3 - 6*(self.y-self.y0)*self.scale**5 + 35*(self.x-self.x0)*self.scale**4*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 70*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 7*(self.y-self.y0)*self.scale**5*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)

	def z40(self,y,x):
		return 15*(self.x-self.x0)*self.scale**4 - 90*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2 + 15*(self.y-self.y0)*self.scale**4 - 42*(self.x-self.x0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 252*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 42*(self.y-self.y0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 28*(self.x-self.x0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 168*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 28*(self.y-self.y0)*self.scale**4*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z41(self,y,x):
		return 60*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale - 60*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**3 - 168*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 168*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 112*(self.x-self.x0)*self.scale**3*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 112*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2

	def z42(self,y,x):
		return -20*(self.x-self.x0)*self.scale**3 + 60*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2 + 105*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 315*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 168*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 504*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 84*(self.x-self.x0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 252*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z43(self,y,x):
		return -60*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale + 20*(self.y-self.y0)*self.scale**3 + 315*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 105*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 504*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 168*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 252*(self.x-self.x0)*self.scale**2*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 84*(self.y-self.y0)*self.scale**3*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3

	def z44(self,y,x):
		return 15*(self.x-self.x0)*self.scale**2 - 15*(self.y-self.y0)*self.scale**2 - 140*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 140*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 420*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 420*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 504*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 504*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 210*(self.x-self.x0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4 - 210*(self.y-self.y0)*self.scale**2*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4

	def z45(self,y,x):
		return 30*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale - 280*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 840*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 1008*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 420*(self.x-self.x0)*self.scale (self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4

	def z46(self,y,x):
		return -6*(self.x-self.x0)*self.scale + 105*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 560*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 1260*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 1260*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4 + 462*(self.x-self.x0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**5

	def z47(self,y,x):
		return -6*(self.y-self.y0)*self.scale + 105*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) - 560*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 + 1260*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 - 1260*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4 + 462*(self.y-self.y0)*self.scale * ((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**5

	def z48(self,y,x):
		return 1 - 42*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2) + 420*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**2 - 1680*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**3 + 3150*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**4 - 2772*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**5 + 924*((self.x-self.x0)*self.scale**2 + (self.y-self.y0)*self.scale**2)**6

