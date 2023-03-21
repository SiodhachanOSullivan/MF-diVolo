import numpy as np
import numba
import scipy.special as sp_spec
import scipy.integrate as sp_int
from scipy.optimize import minimize, curve_fit
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import quad
def pseq_params(params):
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    # for key, dval in zip(['Ntot', 'pconnec', 'gei'], [1, 2., 0.5]):
    #     if key in params.keys():
    #         exec(key+' = params[key]')
    #     else: # default value
    #         exec(key+' = dval')
    Ntot, pconnec, gei = params['Ntot'],params['pconnec'],params['gei']
    if 'P' in params.keys():
        P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10 = params['P']
    else: # no correction
        P0 = -45e-3
        for i in range(1,11):
            exec('P'+str(i)+'= 0')

    return Qe,Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10

# @numba.jit()


def get_fluct_regime_varsup(Fe, Fi, XX,Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    

    
    
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    
    
    
    
    fi = Fi*gei*pconnec*Ntot
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El-XX)/muG
    muGn, Tm = muG/Gl, Cm/muG
    
    Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

    sV = np.sqrt(\
                 fe*(Ue*Te)**2/2./(Te+Tm)+\
                 fi*(Ti*Ui)**2/2./(Ti+Tm))

    fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division,
   
    Tv = ( fe*(Ue*Te)**2 + fi*(Ti*Ui)**2 ) /( fe*(Ue*Te)**2/(Te+Tm) + fi*(Ti*Ui)**2/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV+1e-12, muGn, TvN




def mean_and_var_conductance(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot
    return Qe*Te*fe, Qi*Ti*fi, Qe*np.sqrt(Te*fe/2.), Qi*np.sqrt(Ti*fi/2.)


### FUNCTION, INVERSE FUNCTION
# @numba.jit()
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*(sp_spec.erfc((Vthre-muV)/np.sqrt(2)/sV))

# @numba.jit()
def effective_Vthre(Y, muV, sV, TvN, Gl, Cm):
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff

# @numba.jit()
def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    setting by default to True the square
    because when use by external modules, coeff[5:]=np.zeros(3)
    in the case of a linear threshold
    """
    
    muV0, DmuV0 = -60e-3,10e-3
    sV0, DsV0 =4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    
    return P0+P1*(muV-muV0)/DmuV0+\
        P2*(sV-sV0)/DsV0+P3*(TvN-TvN0)/DTvN0+\
        0*P4*np.log(muGn)+P5*((muV-muV0)/DmuV0)**2+\
        P6*((sV-sV0)/DsV0)**2+P7*((TvN-TvN0)/DTvN0)**2+\
        P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
        P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
        P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0
      
# final transfer function template :
# @numba.jit()

def TF_my_templateup(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    
    

    if(hasattr(fe, "__len__")):
        
        fe[fe<1e-8]=1e-8


    else:
    
        if(fe<1e-8):
            fe=1e-8

    if(hasattr(fi, "__len__")):
    
        fi[fi<1e-8]=1e-8

    
    else:
        
        if(fi<1e-8):
            fi=1e-8

   
   
   
   
   
    muV, sV, muGn, TvN = get_fluct_regime_varsup(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    

    if(hasattr(muV, "__len__")):
        #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sV[sV<1e-4]=1e-4
    
    
    else:
        
        if(sV<1e-4):
            sV=1e-4



    
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    if(hasattr(Fout_th, "__len__")):
        #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th<1e-8]=1e-8
    
    
    else:
        
        if(Fout_th<1e-8):
            Fout_th=1e-8
    '''
    if(El<-0.063):
    
        if(hasattr(Fout_th, "__len__")):
            #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
            Fout_th[Fout_th>80.]=175
    
    
        else:
        
            if(Fout_th>80.):
                print("Done")
                Fout_th=175

    '''
    #print 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',fe,fi,muV, sV, TvN,Fout_th
    return Fout_th



def gaussian(x, mu, sig):
    return (1/(sig*np.sqrt(2*3.1415)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



if __name__=='__main__':

    print(__doc__)
