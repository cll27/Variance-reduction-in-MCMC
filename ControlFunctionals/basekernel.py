#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:24:31 2018

@author: charlinelelan
"""
###########################################################
#                                                         #
# Coded by François-Xavier Briol                          #
#                                                         #
###########################################################

import numpy as np
import autograd.numpy as np

def gradx_kernel_matern12(x,y,l):
    r = np.sqrt(np.sum(np.power((x-y),2)))
    diff = np.atleast_2d(x-y).T
    if r < 0.000001:
        d = len(x)
        out = np.zeros(d)
    else:
        out = -(diff/(l*r))*np.exp(-r/l)
    return(out)
    
def gradxgrady_kernel_matern12(x,y,l):
    r = np.sqrt(np.sum(np.power((x-y),2)))
    diff = np.atleast_2d(x-y).T
    if r < 0.000001:
        d = len(x)
        out = np.zeros((d,d))
    else: 
        diag_term = ((1/(l*r))-(diff**2)*((1./((l**2)*(r**2)))+(1./(l*(r**3)))))*np.exp(-r/l)
        diag_term = diag_term.flatten()
        out = -((1/(l*r**3))+(1/((l**2)*(r**2))))*np.dot(diff,diff.T)*np.exp(-r/l)
        np.fill_diagonal(out,diag_term)
    return(out)
    
def grady_kernel_matern12(x,y,l):
    r = np.sqrt(np.sum(np.power((x-y),2)))
    diff = np.atleast_2d(x-y).T
    if r < 0.000001:
        d = len(x)
        out = np.zeros(d)
    else:
        out = (diff/(l*r))*np.exp(-r/l)
    return(out)
    
def kernel_matern12(x,y,l):
    r = np.sqrt(np.sum(np.power((x-y),2)))
    out = np.exp(-r/l)
    return(out)

class KernelMatern12:

	def __init__(self,kernel_params):
		self.kernel_params = kernel_params

	def __call__(self,x,y):
		return kernel_matern12(x=x,y=y,l=self.kernel_params)

	def gradx(self,x,y):
		return gradx_kernel_matern12(x=x,y=y,l=self.kernel_params)

	def grady(self,x,y):
		return grady_kernel_matern12(x=x,y=y,l=self.kernel_params)

	def gradxgrady(self,x,y):
		return gradxgrady_kernel_matern12(x=x,y=y,l=self.kernel_params)

######################################################################################    
def gradx_kernel_matern32(x,y,l):
    diff = np.atleast_2d(x-y).T
    r = np.sqrt(np.sum(np.power((x-y),2)))
    if r < 0.000001:
        d = len(x)
        out = np.zeros(d)
    else:
        out = -(3.*diff/(l**2))*np.exp(-(np.sqrt(3)*r)/l)
    return(out)
    
def gradxgrady_kernel_matern32(x,y,l):
    diff = np.atleast_2d(x-y).T
    r = np.sqrt(np.sum(np.power((x-y),2)))
    if r < 0.000001:
        d = len(x)
        out = np.zeros((d,d))
    else:
        out = ((-3.*np.sqrt(3.)*np.exp(-np.sqrt(3.)*(r/l)))/(r*l**3))*np.dot(diff,diff.T)
        diag_term = np.exp(-(np.sqrt(3.)*r)/l)*(3./((r**2)*(l**3)))
        diag_term = diag_term*((-np.sqrt(3.)*r*diff**2)+(l*r**2))
        diag_term = diag_term.flatten()
        np.fill_diagonal(out,diag_term)
    return(out)
    
def grady_kernel_matern32(x,y,l):
    diff = np.atleast_2d(x-y).T
    r = np.sqrt(np.sum(np.power((x-y),2)))
    if r < 0.000001:
        d = len(x)
        out = np.zeros(d)
    else:
        out = (3.*diff/(l**2))*np.exp(-(np.sqrt(3)*r)/l)
    return(out)
    
def kernel_matern32(x,y,l):
    r = np.sqrt(np.sum(np.power((x-y),2)))
    out = (1.+(np.sqrt(3)*r)/l)*np.exp(-(np.sqrt(3)*r)/l)
    return(out)
    
class KernelMatern32:

	def __init__(self,kernel_params):
		self.kernel_params = kernel_params

	def __call__(self,x,y):
		return kernel_matern32(x=x,y=y,l=self.kernel_params)

	def gradx(self,x,y):
		return gradx_kernel_matern32(x=x,y=y,l=self.kernel_params)

	def grady(self,x,y):
		return grady_kernel_matern32(x=x,y=y,l=self.kernel_params)

	def gradxgrady(self,x,y):
		return gradxgrady_kernel_matern32(x=x,y=y,l=self.kernel_params)
###################################################################################### 
        
def gradx_kernel_rbf(x,y,l):
	r = np.sum(np.power((x-y),2))
	out = -((x-y)/np.power(l,2))*np.exp(-r/(2.*np.power(l,2)))
	out = np.atleast_2d(out).T
	return(out)
    
def gradxgrady_kernel_rbf(x,y,l):
    d = len(x)
    out = np.empty((d,d))
    r = np.sum(np.power((x-y),2))
    exp_term = np.exp(-r/(2.*l**2))
    diff = np.atleast_2d(x-y).T
    out = -(1./np.power(l,4))*np.dot(diff,diff.T)*exp_term
    np.fill_diagonal(out,((np.power(l,2)-np.power(diff.T,2))/np.power(l,4))*exp_term)
    return(out)
    
def grady_kernel_rbf(x,y,l):
    r = np.sum(np.power((x-y),2))
    out = ((x-y)/np.power(l,2))*np.exp(-r/(2.*np.power(l,2)))
    out = np.atleast_2d(out).T
    return(out)
    
def kernel_rbf(x,y,l):
    r = np.sum(np.power((x-y),2))
    out = np.exp(-r/(2.*np.power(l,2)))
    return(out)
    
class KernelRbf:

	def __init__(self,kernel_params):
		self.kernel_params = kernel_params

	def __call__(self,x,y):
		return kernel_rbf(x=x,y=y,l=self.kernel_params)

	def gradx(self,x,y):
		return gradx_kernel_rbf(x=x,y=y,l=self.kernel_params)

	def grady(self,x,y):
		return grady_kernel_rbf(x=x,y=y,l=self.kernel_params)

	def gradxgrady(self,x,y):
		return gradxgrady_kernel_rbf(x=x,y=y,l=self.kernel_params)
###################################################################################### 
        
def gradx_kernel_invmultiquad(x,y,l):
    r = np.sum(np.power((x-y),2))
    diff = np.atleast_2d(x-y).T
    out = -2.*diff*l*np.power(1.+r,-l-1.)
    return(out)
    
def gradxgrady_kernel_invmultiquad(x,y,l):
    r = np.sum(np.power((x-y),2))
    diff = np.atleast_2d(x-y).T
    out = -4.*l*(l+1.)*np.dot(diff,diff.T)
    out = out*np.power(1.+r,-l-2.)
    diag_term = -4.*l*(l+1.)*np.power(diff,2.)*np.power(1.+r,-l-2.)
    diag_term = diag_term +2.*l*np.power(1.+r,-l-1.)
    np.fill_diagonal(out,diag_term)
    return(out)
    
def grady_kernel_invmultiquad(x,y,l):
    r = np.sum(np.power((x-y),2))
    diff = np.atleast_2d(x-y).T
    out = 2.*diff*l*np.power(1.+r,-l-1.)
    return(out)
    
def kernel_invmultiquad(x,y,l):
    r = np.sum(np.power((x-y),2))
    out = np.power(1.+r,-l)
    return(out)
    
class KernelInvmultiquad:

	def __init__(self,kernel_params):
		self.kernel_params = kernel_params

	def __call__(self,x,y):
		return kernel_invmultiquad(x=x,y=y,l=self.kernel_params)

	def gradx(self,x,y):
		return gradx_kernel_invmultiquad(x=x,y=y,l=self.kernel_params)

	def grady(self,x,y):
		return grady_kernel_invmultiquad(x=x,y=y,l=self.kernel_params)

	def gradxgrady(self,x,y):
		return gradxgrady_kernel_invmultiquad(x=x,y=y,l=self.kernel_params)
    


