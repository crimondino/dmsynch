import math
import numpy as np

# Notebook for user defined units. All dimensional quantities are in GeV = 1
GeV = 1; 
TeV = 10**3 * GeV;
MeV = 10**-3 * GeV; keV = 10**-3 * MeV; eV = 10**-3 * keV; meV = 10**-3 * eV; 

Kg = 5.6096 * 10**26 * GeV
Gram = 10**-3 * Kg
Meter = 1/(0.1973 * 10**-15 * GeV)
CentiMeter = 10**-2 * Meter
FemtoMeter = 10**-15 * Meter;  KiloMeter = 10**3 * Meter;
Second = 2.99792458 * 10**8 * Meter
Hz = Second**-1
kHz = 10**3 * Hz; MHz = 10**6 * Hz; GHz = 10**9 * Hz; THz = 10**12 * Hz; mHz = 10**-3 * Hz;
Hour = 3600*Second
Year = 365*24*Hour

Kelvin = 8.6 * 10**-5 * eV
Joule = Kg * Meter**2 * Second**-2; erg = 10**-7 * Joule; Watt = Joule * Second**-1;
Newton = Kg * Meter * Second**-2; Pa = Kg * Meter**-1 * Second**-2; GPa = 10**9 * Pa;

NAvo = 6.022 * 10**23;
bar = 10**5 * Pa;
Torr = 1/750.061683 * bar;
nst = bar/(273.15 * Kelvin);

MPlanck = 1.2209*math.pow(10, 19)
GN = math.pow(MPlanck, -2)
mPlanck = MPlanck/math.sqrt(8*math.pi)

kpc = 3261*Year;
Mpc = math.pow(10, 3)*kpc
pc = math.pow(10, -3)*kpc
AU = 150 * 10**6 * 10**3 * Meter;
RSolar = 695.51 * 10**6 * Meter; MSolar = 1.98 * 10**30 * Kg; LumSolar = 3.827 * 10**26 * Watt;
MassEarth = 5.972 * 10**24 * Kg;
MassMoon = 0.012300 * MassEarth; 
MassJupiter = 1.898 * 10**27 * Kg; aJupiter = 5.2044 * AU;
MassVenus = 4.867 * 10**24 * Kg; aVenus = 0.72333 * AU;
MassDensityEarth = 5.59 * Kg/(10 * CentiMeter)**3;
RadiusEarth = 6371 * 10**3 * Meter;

Hubble0 = 67.8*math.pow(10, 3)*Meter/Second/Mpc
zeq = 3250
HubbleEq = Hubble0*math.pow(1+zeq, 3/2) 
aeq = 1/(1 + zeq)
RhoCrit = (3*math.pow(Hubble0,2))/(8*math.pi*GN)
hubble0 = Hubble0/(100*10000*Meter/Second/Mpc)
RhoDMU = 0.23*3/(8*math.pi)*math.pow(Hubble0, 2)*math.pow(MPlanck, 2)
RhoDMG = 0.4/math.pow(CentiMeter, 3)
v0DM = 235*1000*Meter/Second
SigmavDM = v0DM/math.sqrt(2)

arcmin = (2*math.pi)/360/60
arcsec = (2*math.pi)/360/3600
mas = math.pow(10, -3)*arcsec; 
muas = math.pow(10, -6)*arcsec;
masy = math.pow(10, -3)*arcsec/Year
muasy = math.pow(10, -6)*arcsec/Year
muasyy = math.pow(10, -6)*arcsec/math.pow(Year, 2)
degree = math.radians(1.0)

Tesla = 195 * eV**2; Gauss = 10**-4 * Tesla;
AlphaEM = 1/137;
ElectronCharge  = np.sqrt(4 * np.pi * AlphaEM); Coulomb = (5.28 * 10**-19)**-1;
Volt = eV/ElectronCharge;
Ampere = Coulomb/Second; Ohm = Volt/Ampere; Farad = Coulomb/Volt; 
Henry = Joule/Ampere**2;
Debye = 3.33564 * 10**-30 * Coulomb * Meter;

MProton = 0.938 * GeV; MPion = 134.976 * MeV;
MElectron = 511. * keV;
MQuarkUp = 2.01 * MeV;
MQuarkDown = 4.79 * MeV;
MHiggs = 125 * GeV; vEW = 246 * GeV;
MuNuclear = ElectronCharge/(2 *MProton); MuBohr = ElectronCharge/(2 *MElectron); BohrRadius = 1/(AlphaEM *MElectron);
Angstrom  = 10**-10 * Meter;
barn = 10**-28 * Meter**2; fb = 10**-15 * barn;
GF = 1.166 * 10**-5 * GeV**-2;

ThetaAxion = 4.1 * 10**-19

Jy = 1e-26*Watt/Meter**2/Hz;