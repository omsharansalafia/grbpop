import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma
from astropy.cosmology import Planck15 as cosmo
from .structjet import ell
from .structjet import eta
from . import pflux
from .pdet import pdet_GBM,pdet_GW170817,pdet_GW_O4,pdet_GW_O3
from .globals import *

z0 = np.logspace(logzmin,logzmax,1000)
dVdz0 = 4*np.pi*cosmo.differential_comoving_volume(z0).to('Gpc3 sr-1').value
dL0 = cosmo.luminosity_distance(z0).to('cm').value

default_theta_pop = {'jetmodel':'smooth double power law','thc':0.04,'Lc*':5e51,'a_L':4.7,'b_L':1.6,'Epc*':17.7e3,'a_Ep':1.9,'b_Ep':1.1,'thw':1.,'A':3.2,'s_c':1.,'y':-0.3,'a':4.6,'b':5.3,'zp':2.2}


def PEpLthv(L,Ep,thv,theta_pop=default_theta_pop):
    """
    P(Ep,L | thv, lpop) from Eqs. 6 and 7 in Salafia+2023
    """
    
    lLc = theta_pop['Lc*']*ell(thv,theta_pop)
    eEpc = theta_pop['Epc*']*eta(thv,theta_pop)
    TH = np.exp(-(lLc/L)**theta_pop['A'])
    return np.exp(-0.5*(np.log((lLc/L)**theta_pop['y']*Ep/eEpc)/theta_pop['s_c'])**2.)/(Ep*np.sqrt(2.*np.pi*theta_pop['s_c']**2))*theta_pop['A']/(gamma(1.-1./theta_pop['A'])*lLc)*(L/lLc)**(-theta_pop['A'])*TH


def PEpL(L,Ep,theta_pop=default_theta_pop,grid=True):
    """
    Returns the probability of the peak energy and luminosity P(E_p, L | theta_pop) conditioned on the population parameters, that is, the population model
    for the intrinsic distributions of these quantities. The theta_pop dictionary must contain information about the jet and shock breakout model, including
    all required parameters.
    """
    th = np.logspace(logthvmin,np.log10(np.pi/2.),1000)
    
    if not (np.isscalar(L) and np.isscalar(Ep)) and grid:
        thg = th.reshape([1,1,len(th)])
        Lg = L.reshape([1,len(L),1])
        Epg = Ep.reshape([len(Ep),1,1])
        
        PEpL_th = PEpLthv(Lg,Epg,thg,theta_pop)
    
        PEpL = np.trapz(PEpL_th*np.sin(thg)*thg,np.log(th),axis=2)
    elif not (np.isscalar(L) and np.isscalar(Ep)):
        Lg = np.expand_dims(L,axis=-1)
        Epg = np.expand_dims(Ep,axis=-1)
        
        Epg,Lg,thg = np.broadcast_arrays(Epg,Lg,th)
        
        PEpL_th = PEpLthv(Lg,Epg,thg,theta_pop)
    
        PEpL = np.trapz(PEpL_th*np.sin(thg)*thg,np.log(th),axis=-1)
    
    else:
        PEpL_th = PEpLthv(L,Ep,th,theta_pop)
    
        PEpL = np.trapz(PEpL_th*np.sin(th)*th,np.log(th),axis=0)
    
    return PEpL

def MD14_SFH(z,a,b,zp):
    """
    Smoothly broken power law event rate density, with functional form given
    in Eq. 9 in Salafia+2023. This is the same functional form as the Madau &
    Dickinson 2014 cosmic star formation history fitting formula.
    """
    return (1.+z)**a/(1.+((1.+z)/(1.+zp))**(b+a))

def Pz(z,theta_pop=default_theta_pop,normalize=True):
    """
    Returns the probability of redshift P(z | theta_pop) conditioned on the population parameters, that is, the population model
    for the intrinsic redshift distribution (as seen by an Earth observer with an infinitely sensitive instrument). The theta_pop dictionary must contain information about the redshift distribution parameters a, b and zp.
    
    If normalize=False, then rho(z)/R0 * 1/(1+z) * dV/dz is returned.
    """
    a = theta_pop['a']
    b = theta_pop['b']
    zp = theta_pop['zp']
    rhoz = MD14_SFH(z,a,b,zp)
    rhoz0 = MD14_SFH(z0,a,b,zp)
    pz = np.interp(z,z0,dVdz0)/(1.+z)*rhoz
    pz0 = dVdz0/(1.+z0)*rhoz0
    if normalize:
        return pz/np.trapz(pz0,z0)
    else:
        return pz
    
    
def obsframe_loglikelihood(pf,ep,alpha=-0.4,specmodel='Comp',pflim=3.5,inst='Fermi',theta_pop=default_theta_pop,res=100,pdet='gbm',return_logalpha=False):
    """
    Loglikelihood contribution from events with unknown redshift. 
    
    Parameters:
    - pf: array of peak fluxes of the event sample, in ph cm-2 s-1, assumed to be in the 50-300 keV band and measured on a 64-ms timescale
    - ep: array of observed peak photon energies, in keV
    - alpha: low-energy spectral index (scalar, mean value of the sample)
    - specmodel: spectral model, either 'Comp' or 'Band'
    - pflim: the sample selection photon flux cut (used only if pdet=None, in which case the sample must be complete in flux!!!)
    - inst: the instrument that collected these data (currently only Fermi and Swift are implemented)
    - theta_pop: dictionary specifying the population parameters
    - res: the grid resolution for integrals
    - pdet: this can be a function that returns the detection probability (float between 0. and 1.) as a function of the 64-ms photon flux in the 50-300 keV band and the observed Epeak. If a string, it can be 'gbm', in which case our result for Fermi/GBM (Salafia & Ravasio 2022) is used. If this is None, then the Pdet is assumed to be 0 below pflim and 1 above: in this case, the sample must be complete in flux.
    - return_logalpha: if True, return the computed value of logalpha (i.e. the logarithm of the integral of Ppop*pdet)
    
    Returns the value of the loglikelihood.
    
    """
    
    
    # construct grid (unequal axes to avoid confusing them)
    L = np.logspace(logLmin,logLmax,res)
    Ep = np.logspace(logEpmin,logEpmax,res+1)
    z = np.logspace(logzmin,logzmax,res-1)
    dL = np.interp(z,z0,dL0)
    
    # make 3D mesh grid
    zg = z.reshape([1,1,len(z)])
    Epg = Ep.reshape([len(Ep),1,1])
    Lg = L.reshape([1,len(L),1])
    
    EpLz = Epg*Lg*zg
    
    # compute population probability distribution
    pz = Pz(z,theta_pop)
    Pepl = PEpL(L,Ep,theta_pop)
    Ppop = Pepl.reshape([len(Ep),len(L),1])*pz
    PpopEpLz = Ppop*EpLz
    
    # compute peak flux on the grid    
    pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,inst=inst,model=specmodel)
    
    # compute ep on the grid
    ep_EpLz = Epg/(1.+zg)
    
    # detection probability
    if pdet is None:
        Pdet = (pf_EpLz>=pflim)
    elif pdet=='gbm':
        Pdet = pdet_GBM(pf_EpLz,ep_EpLz)
    else:
        Pdet = pdet(pf_EpLz,ep_EpLz)
    
    # compute log(fraction of accessible population above the flux limit)
    logalpha = np.log(np.trapz(np.trapz(np.trapz(PpopEpLz*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep),axis=0))
    
    # set up an interpolator of P(Ep,L | theta_pop)
    Itp_logPEpL = RegularGridInterpolator(points=(np.log10(Ep),np.log10(L)),values=np.log10(Pepl),bounds_error=False,fill_value=-np.inf) 
    # start computation of loglikelihood
    logl = 0.
    
    for i in range(len(pf)):
        L_pepz = pflux.L_from_phflux(z,ep[i],pf[i],alpha=alpha,inst=inst,model=specmodel)
        
        logEpL_i = np.reshape((np.log10((1.+z)*ep[i]),np.log10(L_pepz)), (2, -1), order='C').T
        logl_i = np.log(np.trapz(z*(1.+z)*L_pepz/pf[i]*np.nan_to_num(10**Itp_logPEpL(logEpL_i))*pz,np.log(z)))-logalpha
        
        logl += logl_i
    
    # if the result is not finite, return -np.inf
    if np.isfinite(logl):
        if return_logalpha:
            return logl,logalpha
        else:
            return logl
    else:
        if return_logalpha:
            return -np.inf,logalpha
        else:
            return -np.inf


def restframe_loglikelihood(Lobs,Epobs,zobs,alpha=-0.4,specmodel='Comp',pflim=[None,3.5],inst='Fermi+Swift',theta_pop=default_theta_pop,res=100,pdet=None,logalpha=None,prior_EpLz=None,return_logalpha=False):
    """
    Loglikelihood contribution from events with a redshift measurement. 
    
    Parameters:
    - Lobs: array of posterior samples of peak luminosities, in erg/s, shape (N_events,N_samples).
    - Epobs: array of posterior samples of *rest frame* peak photon energies, in keV, shape (N_events,N_samples).
    - zobs: array of posterior samples of redshift, shape (N_events,N_samples).
    - alpha: low-energy spectral index (scalar, mean value of the sample)
    - specmodel: spectral model, either 'Comp' or 'Band'
    - pflim: the sample selection photon flux cut (the sample must be complete in flux above this cut, unless Pdet is given - see below)
    - inst: the instrument that collected these data (currently only Fermi and Swift are implemented)
    - theta_pop: dictionary specifying the population parameters
    - res: the grid resolution for integrals
    - pdet: this can be a function that returns the detection probability (float between 0. and 1.) as a function of the 64-ms photon flux in the 50-300 keV band and the observed Epeak. If a string, it can be 'gbm', in which case our result (Salafia, Ravasio, Ghirlanda & Mandel 2023) is used. If this is None, then the Pdet is assumed to be 0 below pflim and 1 above: in this case, the sample must be complete in flux.
    - logalpha: if given, this is assumed to be the logarithm of the integral of Ppop*pdet, in which case its computation is avoided (to improve performance).
    - prior_EpLz: if given, this is assumed to yield the prior on Ep, L and z, pi(Ep,L,z). If not given, the prior is assumed uniform on both variables. 
    - return_logalpha: if True, return logalpha.
    
    Returns the value of the loglikelihood.
    
    """
    
    # make grid & mesh grid
    L = np.logspace(logLmin,logLmax,res)
    Ep = np.logspace(logEpmin,logEpmax,res+1)
    z = np.logspace(logzmin,logzmax,res-1)
    
    zg = np.copy(z).reshape([1,1,len(z)])
    Epg = np.copy(Ep).reshape([len(Ep),1,1])
    Lg = np.copy(L).reshape([1,len(L),1])
        
    # compute Ppop over the grid, and at the redshifts of the events with a known redshift
    Pepl = PEpL(L,Ep,theta_pop)
        
    # if logalpha is not given, compute it
    if logalpha is None:

        # compute Ppop over the grid
        Ppop = Pepl.reshape([len(Ep),len(L),1])*Pz(z,theta_pop) # full grid
        EpLz = Epg*Lg*zg
        PpopEpLz = Ppop*EpLz
                
        # peak photon flux on the grid & conditioned on the known redshifts
        if inst=='Fermi+Swift':
            pfGBM_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Fermi')
            pfBAT_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst='Swift')
        else:
            pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
        
        ep_EpLz = Epg/(1.+zg)
        
        if pdet is not None:
            if pdet=='gbm':
                if inst=='Fermi+Swift':
                    Pdet = pdet_GBM(pfGBM_EpLz,ep_EpLz)*(pfBAT_EpLz>pflim[1])
                else:
                    Pdet = pdet_GBM(pf_EpLz,ep_EpLz)
            else:
                Pdet = pdet(pf_EpLz,ep_EpLz)
        else:
            if not inst=='Fermi+Swift':
                pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
                Pdet = (pf_EpLz>=pflim)
            else:
                Pdet = (pfGBM_EpLz>=pflim[0])&(pfBAT_EpLz>=pflim[1])
    
        # log(fraction of accessible events)
        logalpha = np.log(np.trapz(np.trapz(np.trapz(PpopEpLz*Pdet,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep),axis=0))
        
        
    # compute loglikelihood
    logl = 0.
    
    # use posterior samples
    Itp_logPEpL = RegularGridInterpolator(points=(np.log10(Ep),np.log10(L)),values=np.log10(Pepl),bounds_error=False,fill_value=-np.inf) # set up an interpolator of P(Ep,L | theta_pop)
    for i in range(Lobs.shape[0]):
        logEpL_i = np.reshape((np.log10(Epobs[i]),np.log10(Lobs[i])), (2, -1), order='C').T # turn posterior samples into an array of (Ep,L) 2D points
        if prior_EpLz is None:
            logl_i = np.log(np.mean(10**Itp_logPEpL(logEpL_i)*Pz(zobs[i],theta_pop))) - logalpha
        else:
            logl_i = np.log(np.mean(10**Itp_logPEpL(logEpL_i)*Pz(zobs[i],theta_pop)/prior_EpLz(Epobs[i],Lobs[i],zobs[i]))) - logalpha
        logl += logl_i
    
    # if the result is not finite, return -np.inf
    if np.isfinite(logl):
        if return_logalpha:
            return logl,logalpha
        else:
            return logl
    else:
        if return_logalpha:
            return -np.inf,logalpha
        else:
            return -np.inf


def known_theta_view_loglikelihood(Ls,Eps,thvs,theta_pop=default_theta_pop,prior_EpLz=None):
    """
    Likelihood of an event with known viewing angle, luminosity and Epeak (e.g. GW170817).
    
    Parameters:
    - Ls: peak luminosity samples, in erg/s. 
    - Eps: *rest frame* peak photon energy samples, in keV.
    - thvs: viewing angle samples, in rad
    - theta_pop: dictionary specifying the population parameters
    - prior_EpLz: if given, this is assumed to yield the prior on Ep, L and z, pi(Ep,L,z). If not given, the prior is assumed uniform on both variables. 
    
    Returns the value of the loglikelihood.
    
    """
    
    # P(L,Ep | theta_pop, theta_view)
    PEpLth = PEpLthv(Ls,Eps,thvs,theta_pop)
    
    # L, Ep prior
    if prior_EpLz is not None:
        prior = prior_EpLz(Eps,Ls,0.)
    else:
        prior = 1.
    
    # Monte Carlo integration 
    like = np.mean(PEpLth/prior)
    
    if np.isfinite(like) and like>0.:
        return np.log(like)
    else:
        return -np.inf

def logalpha_GRB_GW(theta_pop,pflim=3.5,inst='Fermi',pdet_GRB='gbm',pdet_GW='O3',alpha=-0.5,specmodel='Comp',res=60,thvres=300):
    """
    Compute log(alpha), where alpha stands for the integral $ \int P_{pop}(\theta_{source} | \theta_{pop}) P_{det}(\theta_{source}) d\theta_{source} $,
    which represents the normalization of the individual likelihood factors in the hierarchical Bayesian posterior (see Mandel et al. 2019, Salafia et al. 2023).
    Here we assume P_det = P_det_GRB * P_det_GW, i.e. we focus on a multi-messenger detection of a GRB and a GW.  
    
    Parameters:
    - theta_pop: dictionary specifying the population parameters
    - pflim: if pdet_GRB=None, this indicates the photon flux selection cut of the sample
    - inst: either "Fermi" (in which case the photon flux is computed in the 50-300 keV band) or "Swift" (in which case it is computed in the 15-150 keV band) 
    - pdet_GRB: either "gbm" (in which case the GBM detection efficiency model of Salafia et al. 2023 is used), or None, in which case the photon flux cut is used. 
                It can also be a callable, in which case it must be a function of pkflux,E_pobs.
    - pdet_GW: either "O3" (HL sensitivity to BNS mergers in O3, constructed using the injections from https://zenodo.org/record/7890437), "O4" (projected HLV sensitivity to BNS mergers in O4, following Colombo et al. 2022) or "GW170817" (actual HL sensitivity to GW170817, using the psd from https://dcc.ligo.org/LIGO-P1900011/public)
    - res: resolution for L, Ep and z grids
    - thvres: resolution for thv grid
    
    Returns the value of log(alpha) given the population parameter vector theta_pop.
    """

    # construct grids (unequal axes to avoid confusing them)
    thv = np.logspace(logthvmin,np.log10(np.pi/2.),thvres)
    L = np.logspace(logLmin,logLmax,res)
    Ep = np.logspace(logEpmin,logEpmax,res-1)
    z = np.logspace(logzmin,logzmax,res+1)
    
    thvg = thv.reshape([1,1,len(thv),1])
    Lg = L.reshape([1,len(L),1,1])
    Epg = Ep.reshape([len(Ep),1,1,1])
    zg = z.reshape([1,1,1,len(z)])
    
    # P(L,Ep | theta_pop, theta_view)
    PEpL_thv = PEpLthv(Lg,Epg,thvg,theta_pop)
    
    # Pdet_GW
    thvmesh,zmesh = np.broadcast_arrays(thvg,zg)
    if pdet_GW=='GW170817':
        pdetGW_thvz = pdet_GW170817(zmesh.ravel(),thvmesh.ravel()).reshape([1,1,len(thv),len(z)])
    elif pdet_GW=='O4':
        pdetGW_thvz = pdet_GW_O4(zmesh.ravel(),thvmesh.ravel()).reshape([1,1,len(thv),len(z)])
    elif pdet_GW=='O3':
        pdetGW_thvz = pdet_GW_O3(zmesh.ravel(),thvmesh.ravel()).reshape([1,1,len(thv),len(z)])
    else:
        pdetGW_thvz = 1.
    
    # perform integral over thv
    PEpLz_GRBGW = np.trapz(PEpL_thv*pdetGW_thvz*np.sin(thvg)*thvg,np.log(thv),axis=2)*Pz(z,theta_pop).reshape([1,1,len(z)])

    # remove thv axis
    Lg = L.reshape([1,len(L),1])
    Epg = Ep.reshape([len(Ep),1,1])
    zg = z.reshape([1,1,len(z)])
    
    # compute pdet_GRB

    ## peak photon flux and observed Epeak on the grid 
    pf_EpLz = pflux.pflux_from_L(zg,Epg,Lg,alpha=alpha,model=specmodel,inst=inst)
    ep_EpLz = Epg/(1.+zg)

    if pdet_GRB=='gbm':
        Pdet = pdet_GBM(pf_EpLz,ep_EpLz)
    elif pdet_GRB is not None:
        Pdet = pdet_GRB(pf_EpLz,ep_EpLz)
    else:
        Pdet = (pf_EpLz>pflim)
    
    # perform integral over L,Ep,z
    alpha = np.trapz(np.trapz(np.trapz(PEpLz_GRBGW*Pdet*zg*Epg*Lg,np.log(z),axis=2),np.log(L),axis=1),np.log(Ep),axis=0)
    
    return np.log(alpha)


def Plc(x,A):
    """
    Eq. (2) of the paper, where x=Lc/Lc*
    """    
    return A/(math.gamma(1-1/A))*(x)**(-A)*np.exp(-(1/x)**A)


def Pepc(x,sigmac):
    """
    Eq. (3) of the paper, where x=Epc/tilde_Epc
    """
    
    return np.exp(-0.5*((np.log(x))**2/(sigmac))**2)/(np.sqrt(2*np.pi*sigmac**2))
    
