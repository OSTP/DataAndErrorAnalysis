# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 10:56:11 2015

@author: Ryan
"""


import numpy as np
import math


class MCMC:
    """This class is a barebones MCMC implementation; it assumes that it will be 
    given a likelihood function and a function to use to randomize the paramaters.
    The goal of the class is to fill the array of accepted parameters (self.accepted)
    which can then be used for all the posterior definitions"""
    
    def __init__(self,start_pars,par_constraints):
        self.npars=len(start_pars)
        self.start_pars=start_pars
        self.par_constraints=par_constraints
        
    def run(self,nmcmc=1000,burnin=100):
        self.nmcmc=nmcmc
        self.burnin=burnin if burnin<nmcmc else nmcmc/10
   #print ("starting MCMC with"+str(self.nmcmc)+" iterations and "+str(self.burnin)+" burn-in")
        self.current_pars=np.array(self.start_pars)#intial guess for parameters
        self.accepted=np.empty([self.nmcmc,self.current_pars.size]) #keep track of the accepted parameters
        self.acceptanceRatio=0.0#acceptance ratio, to help tune the mcsigmas if needed
        randoms=np.random.uniform(0.,1.,self.nmcmc)
        #Start the MCMC chain
        previousL=self.get_likelihood(self.current_pars,self.par_constraints)
        for step in range(self.nmcmc):
            testPars=self.get_proposal_pars(self.current_pars)#trial parameters
            testL=self.get_likelihood(testPars,self.par_constraints)#trial likelihood
            #Need to handle the case of previousL=0 (since the ratio would be infinite)
            #If during burnin, accept any step, otherwise, warn that this is 
            #probably not a good idea to ignore
            if(previousL==0):#any guess is better than this, although it could wonder way off course!
                r=1#this will force the step to be accepted
                if step>=burnin:
                    print( "WARNING: Accepting random step! Use longer burnin? Better initial guess?")
            else:
                r=min(1,testL/previousL) #Metropolis-Hastings ratio
        
            if randoms[step] <r:#accept this step if true
                    self.current_pars=testPars
                    previousL=testL
                    if step>=self.burnin:
                        self.acceptanceRatio=self.acceptanceRatio+1.
                      
            self.accepted[step]=self.current_pars  
          
        self.acceptanceRatio=self.acceptanceRatio/(self.nmcmc-self.burnin)
       # print ("Done MCMC, acceptance ratio was "+self.acceptanceRatio )


class MCMCResult:
    """This class process the accepted array filled by MCMC"""
    
    def __init__(self,acceptedVals):
        self.accepted=acceptedVals
        self.nacc,self.npars=self.accepted.shape
        self.par_mean=np.empty(self.npars)
        self.par_var=np.empty(self.npars)      
        
    def get_posterior(self,ipar):
        return self.accepted[:,ipar]
    
    def get_par_means(self):
        for i in range(self.npars):
            self.par_mean[i]=np.mean(self.get_posterior(i))
        return self.par_mean
        
    def get_par_variances(self):
        for i in range(self.npars):
            self.par_mean[i]=np.var(self.get_posterior(i))
        return self.par_mean 
        
    def get_autocorr_function(self, ipar, maxlag):
        """Currently INCORRECT!!!!"""
        maxlag=min(maxlag,self.nacc)
        means=self.get_par_means()
        variances=self.get_par_variances()
        
        par_acf=np.empty(maxlag)
        for h in range(maxlag):
            n=0
            d=0
            for i in range(self.nacc-h):
                n=n+(self.accepted[i][ipar]-means[ipar])*(self.accepted[i+h][ipar]-means[ipar])
                #d=d+math.pow(self.accepted[i+h][ipar]-means[ipar],2)
            par_acf[h]=n/variances[ipar]
    
        return par_acf
    
    
    
    
    
    
    
    
    
    
