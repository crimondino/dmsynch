#%%
import numpy as np
import scipy.interpolate as si
#%%

#%%
from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)
MAIN_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(MAIN_DIR)

from my_units import *
#%%

#%%
# scale factor
aa = lambda z: 1./(1.+z)

# black body spectrum
def I0(freq, T):
    x_BB = 2.*np.pi*freq/T
    return 4.*np.pi*freq**3 / (np.exp(x_BB) - 1)

conv_factor = lambda x: (1. - np.exp(-x))/ x
#%%

#%%
#def get_rho_NFW(r_over_rs):
#    '''NFW profile as a function of r/r_s and up to the noermalization rho_s'''
#    return 1/( (r_over_rs) * (1 + r_over_rs)**2 )
#%%

#%%
def get_j0(mchi, sigmav, nu):
    '''Return the dimensionless factor j0, in Jy per Mpc, which is used to multiply j_ell to get the intensity.
    Takes everything in natural units, i.e. mass in GeV, sigmav in 1/GeV^2 and nu in GeV'''
    nu_ref = 1.E11*Hz
    sigmav_ref = 3.E-26*CentiMeter**3/Second
    mchi_ref = 1000
    return 20.8351 * (nu_ref/nu)**1/2 * (sigmav/sigmav_ref) * (mchi_ref/mchi)**2
#%%

#%%
def myFcon(c): return (np.log(1.+c) - (c/(1.+c)))

def my_rhoscale_nfw(mdelta,rdelta,cdelta):
    pref = 1
    rs = rdelta/cdelta
    V = 4.*np.pi * rs**3.
    return pref * mdelta / V / myFcon(cdelta)
#%%

#%%
def get_j_r(Bs, rhos, zs):
    '''Emissivity for DM synchrotron emission'''
    return Bs**(3/2) * (rhos/1e15)**2 / (1+zs)**(9/2)
#%%

#%%
### Linear matter power spectrum for the 2-halo term

def get_fourier_to_multipole_Pkz(zs, ks, chis, ells, Pklin):
    Pzell = np.zeros((len(zs), len(ells)))

    f = si.interp2d(ks, zs, Pklin, bounds_error=True)     
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]
        Pzell[:, ii] = interpolated
    return Pzell
#%%

#%%
### Import the B field profiles
# See Table 1 of 2309.13104 for the halo properties for the given mass bins
file_names = ['profile_bfld_halo_1e10_h12.txt', 'profile_bfld_halo_1e10_h11.txt', 'profile_bfld_halo_1e11_h10.txt', 
              'profile_bfld_halo_1e11_h4.txt', 'profile_bfld_halo_1e12_h12.txt', 'profile_bfld_halo_1e13_h4.txt', 
              'profile_bfld_halo_1e13_h8.txt']
mass_bins = 10**np.array([9.9, 10.4, 10.9, 11.4, 12, 12.5, 13])

# Radial bins are the same for all of the files
rad_bins = np.genfromtxt(MAIN_DIR + '/data/bfield_profiles/'+file_names[0], skip_header=3, max_rows=1)
rad_bins_c = rad_bins[:-1]+(rad_bins[1:]-rad_bins[:-1])/2

Bfiled_grid = np.zeros((len(mass_bins), 66, 23))
logB_interp_list = []

for i, file in enumerate(file_names):
    # Magnetic field profiles in muG
    Bfiled_grid[i] = np.genfromtxt(MAIN_DIR +'/data/bfield_profiles/'+file_names[i], skip_header=7).astype(float)
#    logB_interp_list.append(si.RegularGridInterpolator((np.log10(Bfiled_grid[i][:, 0]), rad_bins_c), np.log10(Bfiled_grid[i][:, 3:]*1E6),
#                                                    bounds_error=False, fill_value=-10 ))
    # Use only every other 8 redshift bins to smooth out the profiles
    logB_interp_list.append(si.RegularGridInterpolator((np.log10(np.concatenate( (Bfiled_grid[i][::8, 0], Bfiled_grid[i][-1:, 0]) )), rad_bins_c),
                                                       np.log10(np.concatenate( (Bfiled_grid[i][::8, 3:], Bfiled_grid[i][-1:, 3:]) )*1E6),
                                                     bounds_error=False, fill_value=-10 )) 
    
def get_pth_profile(rs, zs, m200, r200, rhocritz):  # Eq. (17) from ACT https://arxiv.org/pdf/2009.05558.pdf
    xct = 0.497 * (m200/1e14)**(-0.00865) * (1.+zs)**0.731
    x = rs/r200
    gammat = -0.3
    P0 = 2
    alphat = 0.8
    betat = 2.6
    fb = 0.044/0.25
    
    PGNFW = P0 * (x/xct)**gammat * ( 1 + (x/xct)**alphat )**(-betat)
    P200 = m200*200*rhocritz*fb/(2*r200)    # this has an additional factor of G in front, but we don't care about the units since this is only for modeling the B field profile
    
    return PGNFW*P200



def get_B_rs(rs, zs, ms, m200c, r200c, rhocritz):
    '''Get the value of the magnetic field at radiai rs'''
    rs_ratio = rs/r200c[:, :, None]
    Brs = np.zeros(rs.shape)

    ms_ind = np.digitize(m200c[0, :], mass_bins)
    ms_ind[ms_ind == len(logB_interp_list)] = len(logB_interp_list)-1.

    for i_m in range(len(ms)):
        for i_z, z_val in enumerate(zs):  
            for i_r, rratio_val in enumerate(rs_ratio[i_z, i_m, :]):      
                if rratio_val < rad_bins[0]:
                    #Brs[i_z, i_m, i_r] = ( get_pth_profile(rratio_val*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) /
                    #                       get_pth_profile(rad_bins[0]*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) ) * 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
                    Brs[i_z, i_m, i_r] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
                else:
                    Brs[i_z, i_m, i_r] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rratio_val ] )     
    return Brs
#%%
