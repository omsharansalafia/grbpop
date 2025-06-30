import numpy as np
from .pflux import pflux_from_L
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import os
import pathlib
here = pathlib.Path(__file__).parent.resolve()

# load pdet_GBM grid and create interpolator
ep_grid = np.load(os.path.join(here,'pdet_grid/pdet_ep_grid.npy'))
p64_grid = np.load(os.path.join(here,'pdet_grid/pdet_p64_grid.npy'))
pdet_ep_p64 = np.load(os.path.join(here,'pdet_grid/pdet_ep_p64.npy')) 

Itp_pdet_ep_p64 = RegularGridInterpolator(points=(np.log10(ep_grid),np.log10(p64_grid)),values=np.nan_to_num(np.log10(pdet_ep_p64)),bounds_error=False,fill_value=-100.)


# load pdet_GW grids and create interpolators
z_grid_GWO4 = np.loadtxt(os.path.join(here,'GW_det_eff/z_grid_O4_Colombo22.txt'))
thv_grid_GWO4 = np.loadtxt(os.path.join(here,'GW_det_eff/thv_grid_O4_Colombo22.txt'))
pdetGW_O4 = np.loadtxt(os.path.join(here,'GW_det_eff/GW_det_eff_O4_Colombo22.txt'))

Itp_pdetGW_O4 = RegularGridInterpolator(points=(z_grid_GWO4,thv_grid_GWO4),values=gaussian_filter(pdetGW_O4,0.2),bounds_error=False,fill_value=0.)

z_grid_GWO3 = np.load(os.path.join(here,'GW_det_eff/z_grid_O3.npy'))
thv_grid_GWO3 = np.load(os.path.join(here,'GW_det_eff/thv_grid_O3.npy'))
pdetGW_O3 = np.load(os.path.join(here,'GW_det_eff/GW_det_eff_O3.npy'))

Itp_pdetGW_O3 = RegularGridInterpolator(points=(z_grid_GWO3,thv_grid_GWO3),values=pdetGW_O3,bounds_error=False,fill_value=0.)

z_grid_GW17 = np.loadtxt(os.path.join(here,'GW_det_eff/z_grid_GW17.txt'))
thv_grid_GW17 = np.loadtxt(os.path.join(here,'GW_det_eff/thv_grid_GW17.txt'))
pdetGW_17 = np.loadtxt(os.path.join(here,'GW_det_eff/GW_det_eff_GW17.txt'))

Itp_pdetGW_17 = RegularGridInterpolator(points=(z_grid_GW17,thv_grid_GW17),values=gaussian_filter(pdetGW_17,1.),bounds_error=False,fill_value=0.)


def pdet_GBM(pf,ep):
    """
    Fermi/GBM detection efficiency for short GRBs as a function of the 64-ms photon flux in the 50-300 keV band and the observed epeak. 
    """
    
    logp64 = np.nan_to_num(np.log10(pf))
    logep = np.nan_to_num(np.log10(ep))
    
    logp64 = np.minimum(np.maximum(logp64,np.log10(p64_grid[0])),np.log10(p64_grid[-1]))
    logep = np.minimum(np.maximum(logep,np.log10(ep_grid[0])),np.log10(ep_grid[-1]))
    
    if not np.isscalar(pf):
        epp64_mesh = np.broadcast_arrays(logep,logp64)
        epp64_list = np.reshape(epp64_mesh, (2, -1), order='C').T
    else:
        epp64_list = np.array([logep,logp64])
    
    pdet = np.nan_to_num(10**Itp_pdet_ep_p64(epp64_list))
    
    if not np.isscalar(pf):
        return pdet.reshape(epp64_mesh[0].shape)
    else:
        return pdet

def pdet_GW_O4(z,thv):
    """
    Projected GW detection efficiency during O4, assuming an HLV network.
    """
    
    zz = np.maximum(z,z_grid_GWO4.min())
    
    return Itp_pdetGW_O4(np.vstack([zz,thv]).T)

def pdet_GW_O3(z,thv):
    """
    Projected GW detection efficiency during O3, assuming an HLV network.
    """
    
    zz = np.maximum(z,z_grid_GWO3.min())
    
    return Itp_pdetGW_O3(np.vstack([zz,thv]).T)

def pdet_GW170817(z,thv):
    """
    GW detection efficiency using the HL effective PSD for GW170817.
    """    
    zz = np.maximum(z,z_grid_GW17.min())
    
    return Itp_pdetGW_17(np.vstack([zz,thv]).T)


    
    
