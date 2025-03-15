import astropy.units as u
import astropy.constants as const 
import numpy as np

def get_Ptherm_battaglia(r, M200c, R200c, z, rho_critz):
    """
    See appendix of (https://arxiv.org/pdf/2303.08121). Returns thermal 
    pressure profile in Pascals. Note that the units of input variables are 
    chosen to be consistent with hmvec.
    
    Input:
        r - Radius in Mpc
        M200c - M200c in Msun
        R200c - R200c in Mpc
        z     - redshift 
        rho_critz - critical density at redshfit z in Msun/Mpc^3 (could also compute using hmvec and redshift)
    
    Returns
        Ptherm: thermal gas pressure in Pascals
    """

    # Get parameters (Table X of 2303.08121)
    P0 = 18.1*(M200c/1e14)**(0.154)*(1+z)**(-0.758)
    xc = 0.497*(M200c/1e14)**(-0.00865)*(1+z)**(0.731)
    beta_y = 4.35*(M200c/1e14)**(0.0393)*(1+z)**(0.415)

    # Radius
    x = r/R200c # Check this (could be factors of concentration, etc.)
    
    # Compute P_delta
    fb = cosmo_params_hm_vec['ombh2']/(cosmo_params_hm_vec['ombh2']+cosmo_params_hm_vec['omch2']) # Baryon fraction

    # Gravitational constant with unit conversion so final result is in pascals.
    G = 2.9108200886724906e-40 # Computed from (const.G*u.Msun*(u.Msun/u.Mpc**3)/u.Mpc).to(u.Pa).value
    P_delta = G*200*M200c*rho_critz/(2*R200c)*fb # Dimensions of pressure (Pascals)
    
    # Compute electron pressure
    gamma = -0.3
    Ptherm = P_delta*P0*(x/xc)**(gamma)*(1+(x/xc)**(1))**(-beta_y)
    
    return Ptherm


def get_B_res_beta(z, Rres, M200c, R200c, rho_critz, beta=10,
                   r_R200c_min=0.1, r_R200c_max=1.1):
    """
    Computes magnetic field from constant beta profile using the B12 thermal
    pressure profile. 
    
    Input:
        z     - redshift 
        Rres  - resonant radius in Mpc
        M200c - M200c in Msun
        R200c - R200c in Mpc
        rho_critz - critical density at redshfit z in Msun/Mpc^3 
        beta      - value of plasma beta
        bound_low/bound_high (whether or not to )
        r_R200c_min - minimum radius at which we consider conversion in units of 
                      R200c. Default value is 0.1
        r_R200c_max - maximum radius at which we consider conversion in units of 
                      R200c. Default value is 1.1 which is beyond the truncation
                      scale of 1.08 R200c in the K22 HOD (i.e. doesn't change 
                      result for K22 HOD)
    
    Returns
        B_res: magnetic field value in GeV=1 units
    """
    
    M200c = min(M200c, 10**14.5)
    

    if Rres/R200c<r_R200c_min:
        return 0.0

    if Rres/R200c>r_R200c_max: 
        return 0.0
    
    # Compute thermal pressure using B12 profile (in Pascals)
    Ptherm = get_Ptherm_battaglia(Rres, M200c, R200c, z, rho_critz)
    
    # Get magnetic field and convert to Gauss
    B = np.sqrt(2*const.mu0*Ptherm*u.Pa/beta)
    B_gauss = B.to(u.Gauss).value
    
    # Convert to GeV=1 units
    return B_gauss*1.95e-20
