import math
import numpy as np
from scipy.integrate import quad 
from .structjet import ell
from .structjet import eta
from .pflux import pflux_from_L

def custom_cdf(x,probability_function):
    cdf = np.zeros_like(x)
    for i, val in enumerate(x):
        integral, _ = quad(probability_function, 0.001, val)
        cdf[i] = integral
    return cdf

def extract(probability_function, nsamples):

    x_values = np.logspace(np.log10(0.001), np.log10(1e3), num=int(1e2))  # Adjust the range as needed

    cdf_values = custom_cdf(x_values,probability_function)
    
    random_numbers = np.random.rand(nsamples)
    
    return np.interp(random_numbers, cdf_values, x_values)



def p_lc(x,A):
    
    return A/(math.gamma(1-1/A))*(x)**(-A)*np.exp(-(1/x)**A)


def p_ep(x,sigmac):
    
    return np.exp(-0.5*((np.log(x))**2/(sigmac))**2)/(np.sqrt(2*np.pi*sigmac**2))



    
def extract_lc_ep(lc_star,A,epc_star,y,sigmac,nsamples):
    
    integral_func = lambda x: p_lc(x,A)
    
    l = extract(integral_func,nsamples)*lc_star
    
    
    tilde_e = epc_star *(l/lc_star)**y
    
    
    integral_func = lambda x: p_ep(x,sigmac)
    
    ep = extract(integral_func,nsamples)*tilde_e
    
    return l, ep



def p_flux_samples(z,th,x,plim,alpha,spe_model,inst,num):

    

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


        tildeL = l*ell(th,theta_pop)
        tildeEp = ep*eta(th,theta_pop)
        
        l_samples.extend(tildeL)
        ep_samples.extend(tildeEp)

    p_flux = []
    for n in range(len(l_samples)):
        p_flux.append(pflux_from_L(z,ep_samples[n],l_samples[n],alpha=alpha,model=spe_model,inst=inst))

    p_flux = np.array(p_flux)
    det_prob = np.sum(p_flux > plim)/len(p_flux)

    return p_flux, det_prob
