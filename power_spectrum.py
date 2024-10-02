#%%
import numpy as np
import importlib
import time
import sys
from tqdm import tqdm
from scipy.special import eval_legendre, legendre, spherical_jn

import hmvec as hm
from utils.my_units import *

importlib.reload(sys.modules['utils.power_spectrum_tools'])
from utils.power_spectrum_tools import *
#%%

################### UNITS ###################
# The units from hmvec are:
# proper radius r is always in Mpc
# comoving momentum k is always in Mpc-1
# masses m are in Msolar
# rho densities are in Msolar/Mpc^3
# No h units anywhere

#%%
# Construct the Halo Model using hmvec
# Redshift, mass, and wavenumbers ranges
zMin, zMax, nZs = 0.005, 1.9, 98   #0.005, 1.9, 100  
mMin, mMax, nMs = 1.e10, 1.e16, 102 #1e10, 1e17, 100
kMin, kMax, nKs = 1.e-4, 1.e3, 1001
ms  = np.geomspace(mMin, mMax, nMs)     
zs  = np.linspace(zMin, zMax, nZs)      
ks  = np.geomspace(1e-4, 1e3, nKs)      

hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir')
#%%

#%%
# Useful halo model quantities:
chis     = hcos.comoving_radial_distance(zs)
rvirs    = np.asarray([hcos.rvir(ms, zz) for zz in zs])
cs       = hcos.concentration()
Hubble   = hcos.h_of_z(zs)
nzm      = hcos.get_nzm()
biases   = hcos.get_bh()
deltav   = hcos.deltav(zs)
rhocritz = hcos.rho_critical_z(zs)
PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)

dvols  = chis**2. / Hubble
aas = aa(zs)
rs_nfw = rvirs/cs # scale radius for NFW profile
### check issue with pref in hmvec
#rhos_nfw = hm.rhoscale_nfw(ms, rvirs, cs) # scale density for NFW profile
rhos_nfw = my_rhoscale_nfw(ms, rvirs, cs) # scale density for NFW profile
#%%

#%%
#rMin, rMax, nRs = 1.e-6, 5.e1, 21 #100000
#r_list = np.linspace(rMin, rMax, nRs) 
xMin, xMax, nXs = 1.e-3, 10, 1000

xs = np.geomspace(xMin, xMax, nXs) #np.concatenate( ([0], np.logspace(xMin, xMax, nXs)) ) 
rhos = np.zeros((nZs, nMs, nXs))
#Bs = 0.3*np.ones((nZs, nMs, nXs)) #np.zeros((nZs, nMs, nRs))


rs = xs[None, None, :]*rs_nfw[:, :, None]/aas[:, None, None]
rhos = hm.rho_nfw(rs, rhos_nfw[:, :, None], rs_nfw[:, :, None])

def get_200critz(ms, cs, rhocritz, deltav):
    '''Get m200 and r200'''
    delta_rhos1 = deltav*rhocritz
    delta_rhos2 = 200.*rhocritz
    m200critz = hm.mdelta_from_mdelta(ms, cs, delta_rhos1, delta_rhos2)
    r200critz = hm.R_from_M(m200critz, rhocritz[:,None], delta=200.)
    return m200critz, r200critz

m200c, r200c = get_200critz(ms, cs, rhocritz, deltav)
Bs = get_B_rs(rs, zs, ms, m200c, r200c, rhocritz)
Bs_flat = 0.1*np.ones(rhos.shape)
#%%

#%%
ell_s = aas[:, None]*chis[:, None]/rs_nfw
jrs = get_j_r(Bs, rhos, zs[:, None, None])
jrs_Bflat = get_j_r(Bs_flat, rhos, zs[:, None, None])
#%%

#%%
ell_list = np.concatenate( (np.arange(10, 100, 10), np.arange(100, 1e4, 100), ) )
jells = np.zeros((nZs, nMs, len(ell_list)))
jells_Bflat = np.zeros((nZs, nMs, len(ell_list)))

start_time = time.time()
for i_l, ell in enumerate(tqdm(ell_list)):
    arg_sin = (ell + 0.5)*xs[None, None, :]/ell_s[:, :, None]

    integrand = (xs[None, None, :])**2 * np.sin(arg_sin)/arg_sin * jrs
    jells[:, :, i_l] = 4.*np.pi*rs_nfw/ell_s**2 * np.trapz( integrand, xs, axis=2)

    integrand = (xs[None, None, :])**2 * np.sin(arg_sin)/arg_sin * jrs_Bflat
    jells_Bflat[:, :, i_l] = 4.*np.pi*rs_nfw/ell_s**2 * np.trapz( integrand, xs, axis=2)

print("--- %s seconds ---" % (time.time() - start_time))
#%%

#%%
mchi = 30
freq_radio = 140E6*Hz
sigmav = 2.E-26*CentiMeter**3/Second
j0factor = get_j0(mchi, sigmav, freq_radio)
    
integrand = (xs[None, None, :])**2 * jrs
flux_density = j0factor * rs_nfw**3/(aas[:, None]**5 * chis[:, None]**2) * np.trapz( integrand, xs, axis=2)

integrand = (xs[None, None, :])**2 * jrs_Bflat
flux_density_Bflat = j0factor * rs_nfw**3/(aas[:, None]**5 * chis[:, None]**2) * np.trapz( integrand, xs, axis=2)
#%%

#%% 
#### 1-halo term ####
# integrate over z and m to get the 1halo term
flux_cut = 0.005 # 5 mJy
cut_flux_density = np.heaviside(flux_cut/flux_density-1, 0)
int_over_m = np.trapz(nzm[:, :, None] * np.abs(jells)**2 * cut_flux_density[:, :, None], ms, axis=1)
Cell_1h = np.trapz(dvols[:, None] * int_over_m, zs, axis=0)

cut_flux_density_Bflat = np.heaviside(flux_cut/flux_density_Bflat-1, 0)
int_over_m = np.trapz(nzm[:, :, None] * np.abs(jells_Bflat)**2 * cut_flux_density_Bflat[:, :, None], ms, axis=1)
Cell_1h_Bflat = np.trapz(dvols[:, None] * int_over_m, zs, axis=0)
#%%

#%%
#### 2-halo term ####
# linear matter power at ((ell + 0.5)/chi, z)
Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ell_list, PzkLin)

# integrate over z and m to get the 2halo term
int_over_m_bias = np.trapz(nzm[:, :, None] * biases[:, :, None] * jells * cut_flux_density[:, :, None], ms, axis=1)
Cell_2h = np.trapz(dvols[:, None] * int_over_m_bias**2 * Pzell, zs, axis=0)

int_over_m_bias = np.trapz(nzm[:, :, None] * biases[:, :, None] * jells_Bflat * cut_flux_density_Bflat[:, :, None], ms, axis=1)
Cell_2h_Bflat = np.trapz(dvols[:, None] * int_over_m_bias**2 * Pzell, zs, axis=0)
#%%

#%%
### For the CMB
freq = 143E9*Hz
TCMB = 2.725 * Kelvin
x_BB = 2.*np.pi*freq/TCMB

Cell_1h_T = (conv_factor(x_BB) * TCMB/(1.e-6*Kelvin) / (I0(freq, TCMB)/Jy) )**2 * Cell_1h
Cell_2h_T = (conv_factor(x_BB) * TCMB/(1.e-6*Kelvin) / (I0(freq, TCMB)/Jy) )**2 * Cell_2h
#%%

#%%
### For LOFAR
freq_radio = 140E6*Hz
TERB_140MHz = 193 * Kelvin
conv_radio = 0.5/freq_radio**2

Cell_1h_radio = ( conv_radio/Kelvin )**2 * Cell_1h * Jy**2
Cell_2h_radio = ( conv_radio/Kelvin )**2 * Cell_2h * Jy**2

Cell_1h_radio_Bflat = ( conv_radio/Kelvin )**2 * Cell_1h_Bflat * Jy**2
Cell_2h_radio_Bflat = ( conv_radio/Kelvin )**2 * Cell_2h_Bflat * Jy**2
#%%



################### PLOTS ###################

#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.ticker as mticker

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e'%x))
fmt = mticker.FuncFormatter(g)

rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})
#%%

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

count = 50
c1, c2, g1, g2 = 0, 0, 0, 0
col = sns.color_palette("deep", 4) 
for mi, mm in enumerate(ms):
    if mm in ms[:count]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==0 else None)
        zi = 0
        ax.plot(xs, rhos[zi,mi,:], label=lab(zi), alpha=1-c1/count, color=col[0])
        #ax.plot(xs, jrs[zi,mi,:], label=lab(zi), alpha=1-c1/count, color=col[0])
        c1 += 1
        zi = nZs-1
        ax.plot(xs, rhos[zi,mi,:], label=lab(zi), alpha=1-c2/count, color=col[1])
        #ax.plot(xs, jrs[zi,mi,:], label=lab(zi), alpha=1-c2/count, color=col[0])
        c2 += 1
    elif mm in ms[-count:]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==len(ms)-1 else None)
        zi = 0
        ax.plot(xs, rhos[zi,mi,:], label=lab(zi), alpha=g1/count, color=col[2])
        g1 += 1
        zi = nZs-1
        ax.plot(xs, rhos[zi,mi,:], label=lab(zi), alpha=g2/count, color=col[3])
        g2 += 1
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(loc='best'); ax.grid()
#%%

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

count = 50
c1, c2, g1, g2 = 0, 0, 0, 0
col = sns.color_palette("deep", 4) 
for mi, mm in enumerate(ms):
    if mm in ms[:count]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==0 else None)
        zi = 0
        ax.plot(ell_list, nzm[zi, mi, None]*jells[zi,mi,:]**2, label=lab(zi), alpha=1-c1/count, color=col[0])
        c1 += 1
        zi = nZs-1
        ax.plot(ell_list, nzm[zi, mi, None]*jells[zi,mi,:]**2, label=lab(zi), alpha=1-c2/count, color=col[1])
        c2 += 1
    elif mm in ms[-count:]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==len(ms)-1 else None)
        zi = 0
        ax.plot(ell_list, ms[mi, None]*nzm[zi, mi, None]*jells[zi,mi,:]**2, label=lab(zi), alpha=g1/count, color=col[2])
        g1 += 1
        zi = nZs-1
        ax.plot(ell_list, ms[mi, None]*nzm[zi, mi, None]*jells[zi,mi,:]**2, label=lab(zi), alpha=g2/count, color=col[3])
        g2 += 1
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(loc='best'); ax.grid()
#ax.set_ylim(1E-50,1E-28);
ax.set_xlabel(r'$\ell$', fontsize=15); ax.set_ylabel(r'$(j_{\ell})^2 \times dn/dM $', fontsize=15)
#fig.savefig('figs/Cell1h_contributions.pdf', bbox_inches='tight')
#%%



#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

count = 50
c1, c2, g1, g2 = 0, 0, 0, 0
col = sns.color_palette("deep", 4) 
for mi, mm in enumerate(ms):
    if mm in ms[:count]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==0 else None)
        zi = 0
        ax[0].plot(ell_list, jells[zi,mi,:], label=lab(zi), alpha=1-c1/count, color=col[0])
        ax[0].axvline(ell_s[zi,mi], color=col[0], alpha=1-c1/count, linestyle='dashed', linewidth=0.5)
        c1 += 1
        zi = nZs-1
        ax[0].plot(ell_list, jells[zi,mi,:], label=lab(zi), alpha=1-c2/count, color=col[1])
        ax[0].axvline(ell_s[zi,mi], color=col[1], alpha=1-c2/count, linestyle='dashed', linewidth=0.5)
        c2 += 1
    elif mm in ms[-count:]:
        lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==len(ms)-1 else None)
        zi = 0
        ax[0].plot(ell_list, jells[zi,mi,:], label=lab(zi), alpha=g1/count, color=col[2])
        ax[0].axvline(ell_s[zi,mi], color=col[2], alpha=g1/count, linestyle='dashed', linewidth=0.5)
        g1 += 1
        zi = nZs-1
        ax[0].plot(ell_list, jells[zi,mi,:], label=lab(zi), alpha=g2/count, color=col[3])
        ax[0].axvline(ell_s[zi,mi], color=col[3], alpha=g2/count, linestyle='dashed', linewidth=0.5)       
        g2 += 1
ax[0].set_xlim(10,1e7); #ax.set_ylim(1e-8,1e-2)
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].legend(loc='upper right', ncol=1, fontsize=12, frameon=False); ax[0].grid()
ax[0].set_xlabel(r'$\ell$', fontsize=15); ax[0].set_ylabel(r'$j_{\ell} (M, z)$', fontsize=15)
ax[0].set_title(r'2D projection of the emissivity per halo', fontsize=15)

col = sns.color_palette("deep", len(zs[::15])) 
for iz, z in enumerate(zs[::15]): 
    ax[1].plot(ms, nzm[iz, :]*ms, label=r'$z=${}'.format(fmt(z)), color=col[iz])
ax[1].set_xscale('log'); ax[1].set_yscale('log')
#ax[1].set_xlim(1E10, 1E16); ax[1].set_ylim(1E-45, 1E-20)
ax[1].legend(loc='lower left', ncol=1, fontsize=12, frameon=False); ax[1].grid()
ax[1].set_xlabel(r'$M\ [M_{\odot}]$', fontsize=15); ax[1].set_ylabel(r'$dn/d\log M\ [{\rm Mpc}^{-3}]$', fontsize=15)
ax[1].set_title(r'Halo mass function', fontsize=15);
fig.savefig('figs/jells.pdf', bbox_inches='tight')
#%%


#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
col = sns.color_palette("deep", 4) 

mchi = 500
sigmav_list = np.array([3.E-26, 5.E-26, 1.E-25, 1.E-24])*CentiMeter**3/Second
websky_cells = np.loadtxt('data/websky_cell.txt')

for i, sigmav in enumerate(sigmav_list):
    j0factor = get_j0(mchi, sigmav, freq)
    ax.plot(ell_list, ell_list*(ell_list+1)/(2.*np.pi)*(j0factor**2)*(Cell_1h_T + Cell_2h_T), color=col[i], label=r'{}'.format(fmt(sigmav/(CentiMeter**3/Second))))
    ax.plot(ell_list, ell_list*(ell_list+1)/(2.*np.pi)*(j0factor**2)*(Cell_1h_T ), color=col[i], linestyle='dashed')

ax.plot(websky_cells[:, 0], websky_cells[:, 1], color='k', label='websky radio')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(10, 1e4); #ax.set_ylim(1e-4, 10)
ax.legend(loc='upper left', ncol=1, fontsize=10, frameon=False, title=r'$\langle \sigma v \rangle\ [\mathrm{cm}^3/\mathrm{s}]$')
ax.grid()
ax.set_xlabel(r'$\ell$', fontsize=15); ax.set_ylabel(r'$D_{\ell} = \ell(\ell+1)C_{\ell}/2\pi\ [\mu K^2]$', fontsize=15)
ax.set_title(r'Power spectrum, $m_{\chi} = 500$ GeV', fontsize=15)

#fig.savefig('figs/power_spectra.pdf', bbox_inches='tight')
#%%

#%%
# Plot power spectra in the radio
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
col = sns.color_palette("deep", 4) 

mchi = 30
sigmav_list = np.array([2.E-26, 5.E-26, 1.E-25])*CentiMeter**3/Second
lofar_cells = np.loadtxt('data/LOFAR_cell.txt')
freq_radio = 140E6*Hz
conv_radio = 0.5/freq_radio**2
flux_cut = 0.005 # 5 mJy

for i, sigmav in enumerate(sigmav_list):

    mchi = 20
    j0factor = get_j0(mchi, sigmav, freq_radio)
        
    integrand = (xs[None, None, :])**2 * jrs
    flux_density = j0factor * rs_nfw**3/(aas[:, None]**5 * chis[:, None]**2) * np.trapz( integrand, xs, axis=2)
    cut_flux_density = np.heaviside(flux_cut/flux_density-1, 0)

    integrand_Bflat = (xs[None, None, :])**2 * jrs_Bflat
    flux_density_Bflat = j0factor * rs_nfw**3/(aas[:, None]**5 * chis[:, None]**2) * np.trapz( integrand_Bflat, xs, axis=2)
    cut_flux_density_Bflat = np.heaviside(flux_cut/flux_density_Bflat-1, 0)

    #### 1-halo term ####
    int_over_m = np.trapz(nzm[:, :, None] * np.abs(jells)**2 * cut_flux_density[:, :, None], ms, axis=1)
    Cell_1h = np.trapz(dvols[:, None] * int_over_m, zs, axis=0)
    int_over_m = np.trapz(nzm[:, :, None] * np.abs(jells_Bflat)**2 * cut_flux_density_Bflat[:, :, None], ms, axis=1)
    Cell_1h_Bflat = np.trapz(dvols[:, None] * int_over_m, zs, axis=0)
    #### 2-halo term ####
    Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ell_list, PzkLin)
    int_over_m_bias = np.trapz(nzm[:, :, None] * biases[:, :, None] * jells * cut_flux_density[:, :, None], ms, axis=1)
    Cell_2h = np.trapz(dvols[:, None] * int_over_m_bias**2 * Pzell, zs, axis=0)
    int_over_m_bias = np.trapz(nzm[:, :, None] * biases[:, :, None] * jells_Bflat * cut_flux_density_Bflat[:, :, None], ms, axis=1)
    Cell_2h_Bflat = np.trapz(dvols[:, None] * int_over_m_bias**2 * Pzell, zs, axis=0)

    Cell_1h_radio = ( conv_radio/Kelvin )**2 * Cell_1h * Jy**2
    Cell_2h_radio = ( conv_radio/Kelvin )**2 * Cell_2h * Jy**2
    Cell_1h_radio_Bflat = ( conv_radio/Kelvin )**2 * Cell_1h_Bflat * Jy**2
    Cell_2h_radio_Bflat = ( conv_radio/Kelvin )**2 * Cell_2h_Bflat * Jy**2

    ax.plot(ell_list, ell_list*(ell_list+1)/(2.*np.pi)*(j0factor**2)*(Cell_1h_radio + Cell_2h_radio), color=col[i], label=r'{}'.format(fmt(sigmav/(CentiMeter**3/Second))))
    ax.plot(ell_list, ell_list*(ell_list+1)/(2.*np.pi)*(j0factor**2)*(Cell_1h_radio ), color=col[i], linewidth=1)
    ax.plot(ell_list, ell_list*(ell_list+1)/(2.*np.pi)*(j0factor**2)*(Cell_1h_radio_Bflat + Cell_2h_radio_Bflat), color=col[i], linestyle='dashed')

ax.plot(lofar_cells[:, 0], lofar_cells[:, 1], color='k', label='LOFAR')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(100, 1e4); ax.set_ylim(3e-2, 3000)
ax.legend(loc='upper left', ncol=1, fontsize=10, frameon=False, title=r'$\langle \sigma v \rangle\ [\mathrm{cm}^3/\mathrm{s}]$')
ax.grid()
ax.set_xlabel(r'$\ell$', fontsize=15); ax.set_ylabel(r'$D_{\ell} = \ell(\ell+1)C_{\ell}/2\pi\ [K^2]$', fontsize=15)
ax.set_title(r'Power spectrum 140 MHz, $m_{\chi} =$ '+str(mchi)+' GeV, $S < 5$ mJy', fontsize=15);
ax.text(1500, 0.13, 'Auriga B field (solid)')
ax.text(1500, 0.05, r'$B = 0.1\ \mu$B (dashed)')

#fig.savefig('figs/power_spectra_LOFAR.pdf', bbox_inches='tight')
#%%

#%%
# Plot power spectra in the radio
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
col = sns.color_palette("deep", 4) 

mchi = 30
sigmav_list = np.array([2.E-26, 5.E-26, 1.E-25])*CentiMeter**3/Second
lofar_cells = np.loadtxt('data/LOFAR_cell.txt')
freq_radio = 140E6*Hz
conv_radio = 0.5/freq_radio**2
flux_cut = 0.05 # 5 mJy

for i, sigmav in enumerate(sigmav_list):

    mchi = 20
    j0factor = get_j0(mchi, sigmav, freq_radio)
        
    integrand = (xs[None, None, :])**2 * jrs
    flux_density = j0factor * rs_nfw**3/(aas[:, None]**5 * chis[:, None]**2) * np.trapz( integrand, xs, axis=2)
    cut_flux_density = 1 #np.heaviside(flux_cut/flux_density-1, 0)

    #### 1-halo term ####
    int_over_m = np.trapz(nzm[:, :, None] * np.abs(jells)**2 , ms, axis=1)
    Cell_1h = np.trapz(dvols[:, None] * int_over_m, zs, axis=0)
    #### 2-halo term ####
    Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ell_list, PzkLin)
    int_over_m_bias = np.trapz(nzm[:, :, None] * biases[:, :, None] * jells , ms, axis=1)
    Cell_2h = np.trapz(dvols[:, None] * int_over_m_bias**2 * Pzell, zs, axis=0)

    Cell_1h_radio = ( conv_radio/Kelvin )**2 * Cell_1h * Jy**2
    Cell_2h_radio = ( conv_radio/Kelvin )**2 * Cell_2h * Jy**2

    ax.plot(ell_list, (j0factor**2)*(Cell_1h_radio + Cell_2h_radio)/1E-6, color=col[i], label=r'{}'.format(fmt(sigmav/(CentiMeter**3/Second))))
    ax.plot(ell_list, (j0factor**2)*(Cell_1h_radio)/1E-6, color=col[i], linestyle='dashed')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(100, 1e4); ax.set_ylim(10, 1E5)
ax.legend(loc='upper left', ncol=1, fontsize=10, frameon=False, title=r'$\langle \sigma v \rangle\ [\mathrm{cm}^3/\mathrm{s}]$')
ax.grid()
ax.set_xlabel(r'$\ell$', fontsize=15); ax.set_ylabel(r'$D_{\ell} = \ell(\ell+1)C_{\ell}/2\pi\ [K^2]$', fontsize=15)
ax.set_title(r'Power spectrum, $m_{\chi} =$ '+str(mchi)+' GeV, $S < 5$ mJy', fontsize=15);

#fig.savefig('figs/power_spectra_LOFAR.pdf', bbox_inches='tight')
#%%



#%%
rMin, rMax, nRs = 1.e-3, 10.e1, 10000
m_i_test = 35
z_i_test = 0
r_list = np.geomspace(rMin, rMax, nRs) 
rhos_test = my_rhoscale_nfw(ms[m_i_test], rvirs[z_i_test, m_i_test], cs[z_i_test, m_i_test])

rho_nfw_test = hm.rho_nfw(r_list, rhos_test, rvirs[z_i_test, m_i_test]/cs[z_i_test, m_i_test])


fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(r_list, rho_nfw_test, marker='.', color='k', label='NFW')
ax.set_xscale('log'); ax.set_yscale('log')
#%%