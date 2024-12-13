import numpy as np
from .structjet import ell
from .structjet import eta
from .pflux import pflux_from_L
import grbpop 

from .Ppop import Plc
from .Ppop import Pepc

from scipy.integrate import cumtrapz

e1 = grbpop.globals.e1
e2 = grbpop.globals.e2


def extract(probability_function, nsamples):
    """
    This fucntion extracts nsamples from a 
    given probability function using the method 
    of the cumulative distribution
    """

    x_values = np.logspace(np.log10(e1), np.log10(e2), num=int(1e2))  # Adjust the range as needed

    cdf_values = cumtrapz(probability_function(x_values),x_values, initial=0)
    
    cdf_values /= cdf_values[-1]

    random_numbers = np.random.rand(nsamples)
    
    return np.interp(random_numbers, cdf_values, x_values)




    
def extract_lc_ep(lc_star,A,epc_star,y,sigmac,nsamples):

    """
    This fucntion extracts nsamples of Lc and Epc, 
    defined in eqs. (2) and (3) of the paper.
    """

    integral_func = lambda x: Plc(x,A)
    
    l = extract(integral_func,nsamples)*lc_star
    
    
    tilde_e = epc_star *(l/lc_star)**y
    
    
    integral_func = lambda x: Pepc(x,sigmac)
    
    ep = extract(integral_func,nsamples)*tilde_e
    
    return l, ep



def p_flux_samples(z,thv,x,alpha,model,inst,num):

    """
    This fucntion extracts randomly num simulated peak fluxes,
    once redshift, inclination angle, spectral model and instrument 
    are defined. For each extraction from the hyper-parameters,
    the corresponding Lc and Epc are extracted 1000 times. Therefore, 
    the final sample of peak fluxes consists of 1000*num extractions.
    """
    

    l_samples = []
    ep_samples = []

    out_samples = num

    for i in range(out_samples):
        
        theta_pop = {'jetmodel':'smooth double power law',
                'thc':10**x[i,0],
                'Lc*':10**x[i,1],
                'a_L':x[i,2],
                'b_L':x[i,3],
                'Epc*':10**x[i,4],
                'a_Ep':x[i,5],
                'b_Ep':x[i,6],
                'thw':10**x[i,7],
                'A':x[i,8],
                's_c':10**x[i,9],
                'y':x[i,10],
                'a':x[i,11],
                'b':x[i,12],
                'zp':x[i,13]
                }
        
        l,ep = extract_lc_ep(theta_pop['Lc*'],theta_pop['A'],theta_pop['Epc*'],theta_pop['y'],theta_pop['s_c'],1000)


        tildeL = l*ell(thv,theta_pop)
        tildeEp = ep*eta(thv,theta_pop)
        
        l_samples.extend(tildeL)
        ep_samples.extend(tildeEp)

    p_flux = []
    for n in range(len(l_samples)):
        p_flux.append(pflux_from_L(z,ep_samples[n],l_samples[n],alpha=alpha,model=model,inst=inst))

    p_flux = np.array(p_flux)
    # det_prob = np.sum(p_flux > plim)/len(p_flux)

    return p_flux
