# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import scipy.stats  as stats
import random
from scipy.stats import truncnorm
from scipy.stats import norm
import matplotlib.pyplot as plt

class SDT:

    def __init__(self,sensitivity_msp,sensitivity_lsp,discriminability):
        """ It is assumed that the discriminability levels are uniformly distributed. A convention is when I call a probability about the interval, it is the probability of the right interval. Likewise, the probability about the discriminability level will always be the probability of the high discriminability stimulus. """
        self.min_percept = -7
        self.max_percept = 7
        self.sensitivity_msp = sensitivity_msp
        self.sensitivity_lsp = sensitivity_lsp
        self.discriminability = discriminability #vector of levels of evidence discriminability.
        self.percepts = np.linspace(self.min_percept,self.max_percept,1000)

        """ Conditional Percept Densities: f_{participant}_{realization on which the densities is conditioned}_{realization on which the densities is conditioned} """
        self.f_msp_high_right = norm.pdf(self.percepts,self.discriminability[1],1/self.sensitivity_msp)
        self.f_msp_high_left = norm.pdf(self.percepts,-self.discriminability[1],1/self.sensitivity_msp)
        self.f_msp_low_right = norm.pdf(self.percepts, self.discriminability[0],1/self.sensitivity_msp)
        self.f_msp_low_left = norm.pdf(self.percepts,-self.discriminability[0],1/self.sensitivity_msp)
        self.f_lsp_high_right = norm.pdf(self.percepts,self.discriminability[1],1/self.sensitivity_lsp)
        self.f_lsp_high_left = norm.pdf(self.percepts,-self.discriminability[1],1/self.sensitivity_lsp)
        self.f_lsp_low_right = norm.pdf(self.percepts, self.discriminability[0],1/self.sensitivity_lsp)
        self.f_lsp_low_left = norm.pdf(self.percepts,-self.discriminability[0],1/self.sensitivity_lsp)
        self.f_msp_right = .5*self.f_msp_high_right + .5*self.f_msp_low_right
        self.f_lsp_right = .5*self.f_lsp_high_right + .5*self.f_lsp_low_right
        self.f_lsp = .25*self.f_lsp_high_left + .25*self.f_lsp_low_left + .25*self.f_lsp_low_right + .25*self.f_lsp_high_right
        self.f_msp = .25*self.f_msp_high_left + .25*self.f_msp_low_left + .25*self.f_msp_low_right + .25*self.f_msp_high_right
        """ Posterior Probabilities per Percept: pr_{participant}_{variable about which the probability is being given}_{realization on which the probability is being conditioned}."""
        self.pr_msp_interval = (self.f_msp_high_right+self.f_msp_low_right)/( self.f_msp_high_right+self.f_msp_low_right + self.f_msp_high_left+self.f_msp_low_left )
        self.pr_lsp_interval = (self.f_lsp_high_right+self.f_lsp_low_right)/( self.f_lsp_high_right+self.f_lsp_low_right + self.f_lsp_high_left+self.f_lsp_low_left )
        self.pr_msp_discriminability = (self.f_msp_high_right+self.f_msp_high_left)/( self.f_msp_high_right+self.f_msp_high_left + self.f_msp_low_right+self.f_msp_low_left )
        self.pr_lsp_discriminability = (self.f_lsp_high_right+self.f_lsp_high_left)/( self.f_lsp_high_right+self.f_lsp_high_left + self.f_lsp_low_right+self.f_lsp_low_left )
        self.pr_msp_interval_high = self.f_msp_high_right/( self.f_msp_high_right + self.f_msp_high_left )
        self.pr_lsp_interval_high = self.f_lsp_high_right/( self.f_lsp_high_right + self.f_lsp_high_left )
        self.pr_msp_interval_low = self.f_msp_low_right/( self.f_msp_low_right + self.f_msp_low_left )
        self.pr_lsp_interval_low = self.f_lsp_low_right/( self.f_lsp_low_right + self.f_lsp_low_left )
        """ Conditional Percept Distributions: same notations as the densities.  """
        self.F_msp_high_right = norm.cdf(self.percepts,self.discriminability[1],1/self.sensitivity_msp)
        self.F_msp_high_left = norm.cdf(self.percepts,-self.discriminability[1],1/self.sensitivity_msp)
        self.F_msp_low_right = norm.cdf(self.percepts, self.discriminability[0],1/self.sensitivity_msp)
        self.F_msp_low_left = norm.cdf(self.percepts,-self.discriminability[0],1/self.sensitivity_msp)
        self.F_lsp_high_right = norm.cdf(self.percepts,self.discriminability[1],1/self.sensitivity_lsp)
        self.F_lsp_high_left = norm.cdf(self.percepts,-self.discriminability[1],1/self.sensitivity_lsp)
        self.F_lsp_low_right = norm.cdf(self.percepts, self.discriminability[0],1/self.sensitivity_lsp)
        self.F_lsp_low_left = norm.cdf(self.percepts,-self.discriminability[0],1/self.sensitivity_lsp)

    def calibration_curve_interval(self):
        """  Prints the calibration curves of both subjects. Percepts are assumed to be normally distributed around the discriminability level and with a variance proportional to the participant's sensitivity. """
        
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts,self.pr_lsp_interval,color="red", linewidth=2.5,label="LSP")
        plt.plot(self.percepts,self.pr_msp_interval,color="blue", linewidth=2.5,label="MSP")
        plt.plot(self.percepts,self.pr_lsp_interval_high,'r--',color="red", linewidth=2.5)
        plt.plot(self.percepts,self.pr_msp_interval_high,'r--',color="blue", linewidth=2.5)
        plt.plot(self.percepts,self.pr_msp_interval_low,'r--',color="blue", linewidth=2.5)
        plt.plot(self.percepts,self.pr_lsp_interval_low,'r--',color="red", linewidth=2.5)
        plt.legend(loc='lower right', frameon=False)
        plt.plot((self.min_percept, self.max_percept), (.5, .5), 'k-')
        plt.plot((0, 0), (0, 1), 'k-')
        plt.title('Calibration Curves')
        plt.xlabel('Percepts',fontsize=14)
        plt.ylabel('Right Interval Probability',fontsize=14)
        plt.show()


    def interval_vs_discriminability(self):
        """  Prints the calibration curves of both subjects. Percepts are assumed to be normally distributed around the discriminability level and with a variance proportional to the participant's sensitivity. """
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.pr_lsp_interval,self.pr_lsp_discriminability,color="red",linewidth=2.5) 
        plt.plot(self.pr_msp_interval,self.pr_msp_discriminability,color="blue",linewidth=2.5)  
        plt.xlabel('Probability of Right Interval',fontsize=14)
        plt.ylabel('Probability of High Discriminability',fontsize=14)
        plt.show()

    def calibration_curve_discriminability_vs_interval(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts,self.pr_lsp_discriminability,color="red",linewidth=2.5) 
        plt.plot(self.percepts,self.pr_msp_discriminability,color="blue",linewidth=2.5)  
        plt.plot(self.percepts,self.pr_lsp_interval,'r--' ,color="red",linewidth=2.5) 
        plt.plot(self.percepts,self.pr_msp_interval,'r--' ,color="blue",linewidth=2.5)
        plt.xlabel('Percepts',fontsize=14)
        plt.xlim([0,5])
        plt.ylabel('Conditional Probability',fontsize=14)
        plt.show()


    def percept_distributions(self):
        plt.figure(1)
        plt.subplot(221)
        #plt.plot(self.percepts,.5*self.f_lsp_high_right+.5*self.f_lsp_low_right,color="blue",label="Right Interval")
        #plt.plot(self.percepts,.5*self.f_lsp_high_left+.5*self.f_lsp_low_left,color="red",label="Left Interval")
        plt.plot(self.percepts,self.f_lsp_high_right)
        plt.plot(self.percepts,self.f_lsp_low_right)

        plt.subplot(222)
        plt.plot(self.percepts,.5*self.f_lsp_high_right+.5*self.f_lsp_high_left,color="blue",label="High Discriminability")
        plt.plot(self.percepts,.5*self.f_lsp_low_right+.5*self.f_lsp_low_left,color="red",label="Low Discriminability")

        plt.subplot(223)
        plt.plot(self.percepts,.5*self.f_msp_high_right+.5*self.f_msp_low_right,color="blue")
        plt.plot(self.percepts,.5*self.f_msp_high_left+.5*self.f_msp_low_left,color="red")

        plt.subplot(224)
        plt.plot(self.percepts,.5*self.f_msp_high_right+.5*self.f_msp_high_left,color="blue")
        plt.plot(self.percepts,.5*self.f_msp_low_right+.5*self.f_msp_low_left,color="red")
        plt.show()

    def calibration_interval_and_discriminability(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(.5*self.f_msp_right/self.f_msp, .25*self.f_msp_high_right/self.f_msp, color="blue", label="MSP, High Disc.")
        plt.plot(.5*self.f_msp_right/self.f_msp, .25*self.f_msp_low_right/self.f_msp, color="red", label="MSP, Low Disc.")
        plt.plot(.5*self.f_lsp_right/self.f_lsp, .25*self.f_lsp_high_right/self.f_lsp, 'r--', color="blue", label="LSP, High Disc.")
        plt.plot(.5*self.f_lsp_right/self.f_lsp, .25*self.f_lsp_low_right/self.f_lsp, 'r--', color="red", label="LSP, Low Disc.")
        plt.xlabel('Probability of Right Interval', fontsize=12)
        plt.ylabel('Probability of Right Interval and Discriminability')
        plt.legend(loc='top left')
        plt.show()

    def figure1(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts,self.f_msp_low_right, linewidth=2.5, color="blue", label="Right Interval")
        plt.plot(self.percepts,self.f_msp_low_left, linewidth=2.5, color="red", label="Left Interval")
        plt.text(-.64, .08, 'Discriminability',fontsize='small')
        plt.arrow(-.7, .1, 1.4, 0, head_width=0.01)
        plt.arrow(.7, .1, -1.4, 0, head_width=0.01)
        plt.plot([0.7,0.7],[0,.4], '--')
        plt.plot([-0.7,-0.7],[0,.4], '--', color="red")
        plt.plot([0,4],[0,0],':', linewidth=15, color="blue")
        plt.plot([-4,0],[0,0],':', linewidth=15, color="red")
        plt.ylabel("density", fontsize=14)
        plt.xlabel("percepts", fontsize=14)
        plt.title("SDT, Fixed Discriminability")
        plt.xlim([-4,4])
        plt.legend()
        plt.show()


    def figure2(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts,self.f_msp_low_right,linewidth=2.5,color="blue",label="Right Interval")
        plt.plot(self.percepts,self.f_msp_low_left,linewidth=2.5,color="red",label="Left Interval")
        plt.text(-.15, 0.18, 'Sensitivity',fontsize='x-large')
        plt.arrow(-0.2, .2, 2, 0, head_width=0.01)
        plt.arrow(1.6, .2, -2, 0, head_width=0.01)
        plt.plot([0,4],[0,0],':',linewidth=15,color="blue")
        plt.plot([-4,0],[0,0],':',linewidth=15,color="red")
        plt.ylabel("density",fontsize=14)
        plt.xlabel("percepts",fontsize=14)
        plt.title("SDT, Fixed Discriminability")
        plt.xlim([-4,4])
        plt.ylim([0,.45])
        plt.legend()
        plt.show()

    def figure4(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts,self.f_msp_low_right,linewidth=2.5,color="blue",label="Low Discriminability")
        plt.plot(self.percepts,self.f_msp_low_left,linewidth=2.5,color="red")
        plt.plot(self.percepts,self.f_msp_high_right,'--',linewidth=2.5,color="blue",label="High Discriminability")
        plt.plot(self.percepts,self.f_msp_high_left,'--',linewidth=2.5,color="red")
        plt.plot([0,4],[0,0],':',linewidth=15,color="blue")
        plt.plot([-4,0],[0,0],':',linewidth=15,color="red")
        plt.ylabel("density",fontsize=14)
        plt.xlabel("percepts",fontsize=14)
        plt.title("Percept Distributions, High vs. Low Discriminability")
        plt.xlim([-4,4])
        plt.legend()
        plt.show()

    def figure3(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval_low[self.pr_msp_interval_low>.5],color="black", linewidth=2.5)
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval_low[self.pr_msp_interval_low<.5],color="black", linewidth=2.5)
        plt.plot([0,0],[0,1],'--',color="black")
        plt.xlabel("Percepts",fontsize=15)
        plt.ylabel("Confidence",fontsize=15)
        plt.title("Percept-Confidence Curve, Fixed Discriminability")
        plt.ylim([0,1])
        plt.xlim([-4,4])
        plt.show()


    def figure5(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval_low[self.pr_msp_interval_low>.5],color="black", linewidth=2.5,label="Low Discriminability")
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval_low[self.pr_msp_interval_low<.5],color="black", linewidth=2.5)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval_high[self.pr_msp_interval_high>.5], '--' ,color="black", linewidth=2.5,label="High Discriminability")
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval_high[self.pr_msp_interval_high<.5],'--' ,color="black", linewidth=2.5)
        plt.plot([0,0],[0,1],'--',color="black")
        plt.xlabel("Percepts",fontsize=15)
        plt.ylabel("Confidence",fontsize=15)
        plt.title("Percepts-Confidence Curve")
        plt.ylim([0,1])
        plt.xlim([-4,4])
        plt.legend(loc='bottom right')
        plt.show()

    def figure6(self):
        plt.figure(1)
        plt.subplot(111)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval_low[self.pr_msp_interval_low>.5],color="black", linewidth=2.5,label="Low Discriminability")
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval_low[self.pr_msp_interval_low<.5],color="black", linewidth=2.5)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval_high[self.pr_msp_interval_high>.5], '--' ,color="black", linewidth=2.5,label="High Discriminability")
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval_high[self.pr_msp_interval_high<.5],'--' ,color="black", linewidth=2.5)
        plt.plot(self.percepts[self.percepts>0],self.pr_msp_interval[self.pr_msp_interval>.5] ,color="green", linewidth=2.5,label="Mixed Discriminability")
        plt.plot(self.percepts[self.percepts<0],1-self.pr_msp_interval[self.pr_msp_interval<.5] ,color="green", linewidth=2.5)
        plt.plot([0,0],[0,1],'--',color="black")
        plt.xlabel("Percepts",fontsize=15)
        plt.ylabel("Confidence",fontsize=15)
        plt.title("Percepts-Confidence Curve, Mixed Discriminability")
        plt.ylim([0,1])
        plt.xlim([-4,4])
        plt.legend(loc='bottom right')
        plt.show()
        
        
sdt = SDT(1,1,[0.7,1.3])
sdt.figure1()
sdt.figure2()
sdt.figure3()
sdt.figure4()
sdt.figure5()
sdt.figure6()
