"""
This code imports the G10COSMOS data fits for MAGPHYS and CIGALE and compares them.
"""
################################################################################
### Importing modules ##########################################################
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import scipy.interpolate as interp
import pyfits
import sys
import scipy.stats

from mpl_toolkits.axes_grid import AxesGrid, make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

from itertools import compress

from scipy.stats import gaussian_kde

plt.close('all')


################################################################################
### Importing data #############################################################
################################################################################
# Data-path
path = '/home/prongs/c1124553/PhD_Year2/Cosmos_DMF/Cross_matched_data/Minimized_tables/'
# Filenames
photzf = "MAGPHYS_LAMBDAR_CIGALE_photz_1arcsec.csv"
speczf = "MAGPHYS_LAMBDAR_CIGALE_specz_1arcsec.csv"
# Import the files
# Photz or Specz refers to CIGALE, as to whether the redshifts used in CIGALE are
# photometric or spectroscopic
photzfil = pd.read_csv(path+photzf)
speczfil = pd.read_csv(path+speczf)


################################################################################
### Filtering out likely poor MAGPHYS fits based on FIR data ###################
################################################################################
# CIGALE data has already been subjected to pretty good quality checks - CIGALE
# was only run on sources with at least two FIR measurements. I don't think the
# MAGPHYS data was subjected to this sort of cut because there are way more 
# sources for MAGPHYS than CIGALE.
# So, we are going to do some kind of MAGPHYS filtering. For each row of data (i.e.
# each source, we are going to check and make sure that there are at least two
# Herschel fluxes measured as a flux, rather than -9998, which seems to be the
# default if there is no flux measurement. 
# We will ignore, for now, any SNR requirement on the fluxes i.e. the minimum 
# two fluxes must have a SNR of 2. Let's just see how many we filter out this way
# because although the CIGALE data is probably reasonably realiable for these 
# galaxies, we probably cannot trust the MAGPHYS results.

# Start with extracting data
# We are not interested in the errors for now
herschel_lambdar = {}
wavelengths = ['100', '160', '250', '350', '500']
keynames = ['pacs_100', 'pacs_160', 'spire_250_1', 'spire_350_1', 'spire_500_1']
for i in range(0, len(wavelengths)):
	herschel_lambdar['phot'+wavelengths[i]] = photzfil[keynames[i]].values
	herschel_lambdar['spec'+wavelengths[i]] = speczfil[keynames[i]].values

# Now we need to find out, for each source, how many FIR flux measurements they
# have. If it is <2, then we need to flag this source as not reliable
reliable_photz = [] # True = not reliable, apply mask. False = reliable
reliable_specz = [] # True = not reliable, apply mask. False = reliable
for dattype in ['phot', 'spec']: # do photometric then spectroscopic redshifts
	masks = []
	for wave in wavelengths: # loop over each wavelength
		data = herschel_lambdar[dattype+wave] # extract data for a wavelength
		data = np.ma.masked_where(data == -9998, data) # masks values where they == 9998
		datamask = data.mask # extract the mask
		masks.append(datamask) # save the mask
	# Now we need to loop over the masks source by source. If there are 3+ True,
	# then this source is no good and we need to mark it as unusable
	for i in range(0, len(masks[0])): # looping over each source
		flag = 0 # How many bad fluxes for a source?
		for maskk in masks:
			if maskk[i] == True: # For each wavelength, if mask == True, count
				flag = flag+1
		if flag==4 or flag==5 and dattype=='phot': # If there are 4 or 5 True, then source not trusted		
			reliable_photz.append(True) 
		elif flag==4 or flag==5 and dattype=='spec':
			reliable_specz.append(True)
		elif flag<4 and dattype=='phot': # If there are less than 4 True, then source is trusted
			reliable_photz.append(False)
		elif flag<4 and dattype=='spec':
			reliable_specz.append(False)
# Make the mask an array
reliable_photz = np.asarray(reliable_photz)
reliable_specz = np.asarray(reliable_specz)

print('Number masked out (photz): ',np.sum(reliable_photz))
print('Number masked out (specz): ',np.sum(reliable_specz))

################################################################################
### Masking out sources based on redshift differences between mag and cig ######
################################################################################
# We only want to consider sources which have very similar redshifts in terms of
# magphys and cigale. This is because the models used in calculations depend on 
# the redshift used
# Extract the redshift data
zdiff_phot = photzfil['delta_z']
zdiff_spec = speczfil['delta_z']
# Find the ones with large redshift differences and nans
badz_photz = np.logical_or(zdiff_phot > 5E-5, zdiff_phot < (-5E-5)) + np.isnan(zdiff_phot)
badz_specz = np.logical_or(zdiff_spec > 5E-5, zdiff_spec < (-5E-5)) + np.isnan(zdiff_spec)
# Add these on
reliable_photz = reliable_photz + badz_photz
reliable_specz = reliable_specz + badz_specz

print('Number masked out (photz): ',np.sum(reliable_photz))
print('Number masked out (specz): ',np.sum(reliable_specz))

# Now we can mask the data we will compare i.e. we mask out the dust mass, stellar
# mass, sSFH 
# First, extract the data we need
masssun = 1.989E30
mag_params = {}
mag_params['phot_Mstar'] = photzfil['mass_stellar_percentile50'].values # units log[Msun]
mag_params['phot_SFR']  = photzfil['SFR_0_1Gyr_percentile50'].values    # units log[Msun/yr]
mag_params['phot_Mdust'] = photzfil['mass_dust_percentile50'].values    # units log[Msun]
mag_params['spec_Mstar'] = speczfil['mass_stellar_percentile50'].values # units log[Msun]
mag_params['spec_SFR']  = speczfil['SFR_0_1Gyr_percentile50'].values    # units log[Msun/yr]
mag_params['spec_Mdust'] = speczfil['mass_dust_percentile50'].values    # units log[Msun]
cig_params = {}
cig_params['phot_Mstar'] = np.log10(photzfil['UVoptIR_bayes.stellar.m_star'].values)       # units log[Msun]
cig_params['phot_SFR']  = np.log10(photzfil['UVoptIR_bayes.sfh.sfr10Myrs'].values)         # units log[Msun/yr]
cig_params['phot_Mdust'] = np.log10(photzfil['UVoptIR_bayes.dust.mass'].values / masssun)  # units log[Msun]
cig_params['spec_Mstar'] = np.log10(speczfil['UVoptIR_bayes.stellar.m_star'].values)       # units log[Msun]
cig_params['spec_SFR']  = np.log10(speczfil['UVoptIR_bayes.sfh.sfr10Myrs'].values)         # units log[Msun/yr]
cig_params['spec_Mdust'] = np.log10(speczfil['UVoptIR_bayes.dust.mass'].values / masssun)  # units log[Msun]

# Next, mask this data using the masks we made above
phot_params = ['phot_Mstar','phot_SFR','phot_Mdust']
spec_params = ['spec_Mstar','spec_SFR','spec_Mdust']
for i in [0,1,2]:
	mag_params[phot_params[i]] = np.ma.array(mag_params[phot_params[i]], mask=reliable_photz)
	mag_params[spec_params[i]] = np.ma.array(mag_params[spec_params[i]], mask=reliable_specz)
	cig_params[phot_params[i]] = np.ma.array(cig_params[phot_params[i]], mask=reliable_photz)
	cig_params[spec_params[i]] = np.ma.array(cig_params[spec_params[i]], mask=reliable_specz)

# Now everything is in the same units and we have masked the ones which potentially
# have unreliable magphys results, we can make some plots
# Note that the sSFRs are not calculated over the same window, so there is a BIG CAVEAT
# when comparing these two numbers
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = False)
f.tight_layout()
ax1.plot(mag_params['phot_Mstar'], cig_params['phot_Mstar'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax1.plot(mag_params['spec_Mstar'], cig_params['spec_Mstar'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax1.plot(np.linspace(0,15,1000), np.linspace(0,15,1000), 'k-', label='1:1')
ax1.set_xlabel('$log_{10}(M_{*,MAG})$ [$M_{\odot}$]', fontsize=20)
ax1.set_ylabel('$log_{10}(M_{*,CIG})$ [$M_{\odot}$]', fontsize=20)
ax1.legend(loc='best')

ax2.plot(mag_params['phot_SFR'], cig_params['phot_SFR'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax2.plot(mag_params['spec_SFR'], cig_params['spec_SFR'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax2.plot(np.linspace(-10,4,1000), np.linspace(-10,4,1000), 'k-', label='1:1')
ax2.set_xlabel('$log_{10}(SFR_{MAG})$ [$year^{(-1)}$]', fontsize=20)
ax2.set_ylabel('$log_{10}(SFR_{CIG})$ [$year^{(-1)}$]', fontsize=20)
ax2.legend(loc='best')

ax3.plot(mag_params['phot_Mdust'], cig_params['phot_Mdust'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax3.plot(mag_params['spec_Mdust'], cig_params['spec_Mdust'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax3.plot(np.linspace(-4,12,1000), np.linspace(-4,12,1000), 'k-', label='1:1')
ax3.set_xlabel('$log_{10}(M_{d,MAG})$ [$M_{\odot}$]', fontsize=20)
ax3.set_ylabel('$log_{10}(M_{d,CIG})$ [$M_{\odot}$]', fontsize=20)
ax3.legend(loc='best')


################################################################################
### Filtering out suspected poor fits based on initial plots ###################
################################################################################
# The plot above shows some sources which are clustered about a MAGPHYS value of
# logM* ~ 2    logMdust ~ -2     and some sources which are completely offset from
# the rest of the data in parameter space. They are almost all the same sources
# in both plots. The ones clustered at 2 and -2 I suspect to be a fitting fail 
# i.e. a non-value for these galaxies. The ones which are offset I suspect to be
# poor fits. We unfortunately do not have the chi-sq for the fits for MAGPHYS
# so we cannot be 100% sure. 

# We are going to mask out these sources so that we don't have to deal with them
# True = bad source, mask me. False = good source, don't mask me.

# First the dodgy magphys ones
badmagphys_phot = photzfil['mass_stellar_percentile50'].values < 3 #TrueFalse
badmagphys_spec = speczfil['mass_stellar_percentile50'].values < 3 #TrueFalse
# Now the dodgy CIGALE ones, and we can check the chi-sq of these (expect it to 
# be high and bad)
badcigale_spec   = np.logical_and(speczfil['mass_dust_percentile50'].values > 2, np.log10(speczfil['UVoptIR_bayes.dust.mass'].values/1.989E30) < 0) #TrueFalse
chisq_check_dust = np.where(np.logical_and(speczfil['mass_dust_percentile50'].values > 2, np.log10(speczfil['UVoptIR_bayes.dust.mass'].values/1.989E30) < 0))[0].tolist()
# Pull out the chi-sq values
chisq_cig = speczfil['UVoptIR_best.reduced_chi_square'].values[chisq_check_dust]
# Which ones are high?
print(" ")
print("Fraction of those offset galaxies have a chi-sq > 2? "),
print('%.2f' % (sum(chisq_cig > 2)*1.0 / len(chisq_cig)*1.0))
print(" ")
# A good fraction of these do seem to have terrible fits so we want to mask out
# these sources too

# We need to add these bad magphys sources and bad cigale sources to our mask
reliable_photz_extra = reliable_photz + badmagphys_phot 
reliable_specz_extra = reliable_specz + badmagphys_spec + badcigale_spec

# Next, remask the data using the new masks
phot_params = ['phot_Mstar','phot_SFR','phot_Mdust']
spec_params = ['spec_Mstar','spec_SFR','spec_Mdust']
for i in [0,1,2]:
	mag_params[phot_params[i]] = np.ma.array(mag_params[phot_params[i]], mask=reliable_photz_extra)
	mag_params[spec_params[i]] = np.ma.array(mag_params[spec_params[i]], mask=reliable_specz_extra)
	cig_params[phot_params[i]] = np.ma.array(cig_params[phot_params[i]], mask=reliable_photz_extra)
	cig_params[spec_params[i]] = np.ma.array(cig_params[spec_params[i]], mask=reliable_specz_extra)

# Redo the plots
f2, (ax4, ax5, ax6) = plt.subplots(1,3, sharey = False)
f2.tight_layout(rect=(0.0, 0.0, 1, 1))
ax4.plot(mag_params['phot_Mstar'], cig_params['phot_Mstar'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax4.plot(mag_params['spec_Mstar'], cig_params['spec_Mstar'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax4.plot(np.linspace(0,15,1000), np.linspace(0,15,1000), 'k-', label='1:1')
ax4.set_xlabel('$log_{10}(M_{*,MAG})$ [$M_{\odot}$]', fontsize=20)
ax4.set_ylabel('$log_{10}(M_{*,CIG})$ [$M_{\odot}$]', fontsize=20)
ax4.legend(loc='best')

ax5.plot(mag_params['phot_SFR'], cig_params['phot_SFR'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax5.plot(mag_params['spec_SFR'], cig_params['spec_SFR'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax5.plot(np.linspace(-10,4,1000), np.linspace(-10,4,1000), 'k-', label='1:1')
ax5.set_xlabel('$log_{10}(SFR_{MAG})$ [$year^{(-1)}$]', fontsize=20)
ax5.set_ylabel('$log_{10}(SFR_{CIG})$ [$year^{(-1)}$]', fontsize=20)
ax5.legend(loc='best')

ax6.plot(mag_params['phot_Mdust'], cig_params['phot_Mdust'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax6.plot(mag_params['spec_Mdust'], cig_params['spec_Mdust'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax6.plot(np.linspace(-4,12,1000), np.linspace(-4,12,1000), 'k-', label='1:1')
ax6.set_xlabel('$log_{10}(M_{d,MAG})$ [$M_{\odot}$]', fontsize=20)
ax6.set_ylabel('$log_{10}(M_{d,CIG})$ [$M_{\odot}$]', fontsize=20)
ax6.legend(loc='best')

"""
# Fitting straight line to the data (three lines, the individual z and total)
# Just photz
line_mstar_phot = np.ma.polyfit(mag_params['phot_Mstar'], cig_params['phot_Mstar'],1)
lp_mstar_phot   = np.poly1d(line_mstar_phot)

line_sSFR_phot  = np.ma.polyfit(mag_params['phot_sSFR'], cig_params['phot_sSFR'],1)
lp_sSFR_phot    = np.poly1d(line_sSFR_phot)

line_mdust_phot = np.ma.polyfit(mag_params['phot_Mdust'], cig_params['phot_Mdust'],1)
lp_mdust_phot   = np.poly1d(line_mdust_phot)

# Just specz
line_mstar_spec = np.ma.polyfit(mag_params['spec_Mstar'], cig_params['spec_Mstar'],1)
lp_mstar_spec   = np.poly1d(line_mstar_spec)

line_sSFR_spec  = np.ma.polyfit(mag_params['spec_sSFR'], cig_params['spec_sSFR'],1)
lp_sSFR_spec    = np.poly1d(line_sSFR_spec)

line_mdust_spec = np.ma.polyfit(mag_params['spec_Mdust'], cig_params['spec_Mdust'],1)
lp_mdust_spec   = np.poly1d(line_mdust_spec)

# Both of them together (need to join data)
mag_mstar_concat = np.ma.concatenate((mag_params['phot_Mstar'], mag_params['spec_Mstar']))
cig_mstar_concat = np.ma.concatenate((cig_params['phot_Mstar'], cig_params['spec_Mstar']))
mag_sSFR_concat  = np.ma.concatenate((mag_params['phot_sSFR'], mag_params['spec_sSFR']))
cig_sSFR_concat  = np.ma.concatenate((cig_params['phot_sSFR'], cig_params['spec_sSFR']))
mag_mdust_concat = np.ma.concatenate((mag_params['phot_Mdust'], mag_params['spec_Mdust']))
cig_mdust_concat = np.ma.concatenate((cig_params['phot_Mdust'], cig_params['spec_Mdust']))

line_mstar_concat = np.ma.polyfit(mag_mstar_concat, cig_mstar_concat,1)
lp_mstar_concat   = np.poly1d(line_mstar_concat)

line_sSFR_concat = np.ma.polyfit(mag_sSFR_concat, cig_sSFR_concat,1)
lp_sSFR_concat   = np.poly1d(line_sSFR_concat)

line_mdust_concat = np.ma.polyfit(mag_mdust_concat, cig_mdust_concat,1)
lp_mdust_concat   = np.poly1d(line_mdust_concat)
"""

# With the axes limits
f3, (ax7, ax8, ax9) = plt.subplots(1,3, sharey = False)
f3.tight_layout(rect=(0.0, 0.0, 1, 1))
ax7.plot(mag_params['phot_Mstar'], cig_params['phot_Mstar'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax7.plot(mag_params['spec_Mstar'], cig_params['spec_Mstar'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax7.plot(np.linspace(4,13,1000), np.linspace(4,13,1000), 'k-', label='1:1')
#ax7.plot(np.linspace(4,13,1000), lp_mstar_phot(np.linspace(4,13,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                    # Photz sources fit
#ax7.plot(np.linspace(4,13,1000), lp_mstar_spec(np.linspace(4,13,1000)), '-', color='purple', label='Specz fit', linewidth=2)                    # Specz sources fit
#ax7.plot(np.linspace(4,13,1000), lp_mstar_concat(np.linspace(4,13,1000)), '-', color = 'grey', label='Total fit', linewidth=2) # All sources fit
ax7.set_xlabel('$log_{10}(M_{*,MAG})$ [$M_{\odot}$]', fontsize=20)
ax7.set_ylabel('$log_{10}(M_{*,CIG})$ [$M_{\odot}$]', fontsize=20)
ax7.legend(loc='best')
ax7.set_xlim(4,14)
ax7.set_ylim(7,13)

ax8.plot(mag_params['phot_SFR'], cig_params['phot_SFR'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax8.plot(mag_params['spec_SFR'], cig_params['spec_SFR'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax8.plot(np.linspace(-5,6,1000), np.linspace(-5,6,1000), 'k-', label='1:1')
#ax8.plot(np.linspace(-13.5,-7,1000), lp_sSFR_phot(np.linspace(-13.5,-7,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                     # Photz sources fit
#ax8.plot(np.linspace(-13.5,-7,1000), lp_sSFR_spec(np.linspace(-13.5,-7,1000)), '-', color='purple', label='Specz fit', linewidth=2)                     # Specz sources fit
#ax8.plot(np.linspace(-13.5,-7,1000), lp_sSFR_concat(np.linspace(-13.5,-7,1000)), '-', color = 'grey', label='Total fit', linewidth=2)  # All sources fit
ax8.set_xlabel('$log_{10}(SFR_{MAG})$ [$year^{(-1)}$]', fontsize=20)
ax8.set_ylabel('$log_{10}(SFR_{CIG})$ [$year^{(-1)}$]', fontsize=20)
ax8.legend(loc='best')
ax8.set_xlim(-5,5)
ax8.set_ylim(-4,6)

ax9.plot(mag_params['phot_Mdust'], cig_params['phot_Mdust'], 'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax9.plot(mag_params['spec_Mdust'], cig_params['spec_Mdust'], 'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax9.plot(np.linspace(2,12,1000), np.linspace(2,12,1000), 'k-', label='1:1')
#ax9.plot(np.linspace(0,12,1000), lp_mdust_phot(np.linspace(0,12,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                     # Photz sources fit
#ax9.plot(np.linspace(0,12,1000), lp_mdust_spec(np.linspace(0,12,1000)), '-', color='purple', label='Specz fit', linewidth=2)                     # Specz sources fit
#ax9.plot(np.linspace(0,12,1000), lp_mdust_concat(np.linspace(0,12,1000)), '-', color = 'grey', label='Total fit', linewidth=2)  # All sources fit
ax9.set_xlabel('$log_{10}(M_{d,MAG})$ [$M_{\odot}$]', fontsize=20)
ax9.set_ylabel('$log_{10}(M_{d,CIG})$ [$M_{\odot}$]', fontsize=20)
ax9.legend(loc='best')
ax9.set_xlim(2,12)
ax9.set_ylim(4,12)


################################################################################
### Doing the flat comparison plots (1:1 line is flat) #########################
################################################################################
# This is not the best way to plot the data 
# The best way is with the 1:1 line being flat - will make these plots here
# We also want to fit straight lines to this data to see how it compares to the 
# 1:1 line

# Fitting straight line to the data (three lines, the individual z and total)
# Just photz
line_mstar_phot = np.ma.polyfit(mag_params['phot_Mstar'], cig_params['phot_Mstar'] - mag_params['phot_Mstar'],1)
lp_mstar_phot   = np.poly1d(line_mstar_phot)

line_sSFR_phot  = np.ma.polyfit(mag_params['phot_SFR'], cig_params['phot_SFR'] - mag_params['phot_SFR'],1)
lp_sSFR_phot    = np.poly1d(line_sSFR_phot)

line_mdust_phot = np.ma.polyfit(mag_params['phot_Mdust'], cig_params['phot_Mdust'] - mag_params['phot_Mdust'],1)
lp_mdust_phot   = np.poly1d(line_mdust_phot)

# Just specz
line_mstar_spec = np.ma.polyfit(mag_params['spec_Mstar'], cig_params['spec_Mstar'] - mag_params['spec_Mstar'],1)
lp_mstar_spec   = np.poly1d(line_mstar_spec)

line_sSFR_spec  = np.ma.polyfit(mag_params['spec_SFR'], cig_params['spec_SFR'] - mag_params['spec_SFR'],1)
lp_sSFR_spec    = np.poly1d(line_sSFR_spec)

line_mdust_spec = np.ma.polyfit(mag_params['spec_Mdust'], cig_params['spec_Mdust'] - mag_params['spec_Mdust'],1)
lp_mdust_spec   = np.poly1d(line_mdust_spec)

# Both of them together (need to join data)
mag_mstar_concat = np.ma.concatenate((mag_params['phot_Mstar'], mag_params['spec_Mstar']))
cig_mstar_concat = np.ma.concatenate((cig_params['phot_Mstar'], cig_params['spec_Mstar']))
mag_sSFR_concat  = np.ma.concatenate((mag_params['phot_SFR'], mag_params['spec_SFR']))
cig_sSFR_concat  = np.ma.concatenate((cig_params['phot_SFR'], cig_params['spec_SFR']))
mag_mdust_concat = np.ma.concatenate((mag_params['phot_Mdust'], mag_params['spec_Mdust']))
cig_mdust_concat = np.ma.concatenate((cig_params['phot_Mdust'], cig_params['spec_Mdust']))

line_mstar_concat = np.ma.polyfit(mag_mstar_concat, cig_mstar_concat - mag_mstar_concat,1)
lp_mstar_concat   = np.poly1d(line_mstar_concat)

line_sSFR_concat = np.ma.polyfit(mag_sSFR_concat, cig_sSFR_concat - mag_sSFR_concat,1)
lp_sSFR_concat   = np.poly1d(line_sSFR_concat)

line_mdust_concat = np.ma.polyfit(mag_mdust_concat, cig_mdust_concat - mag_mdust_concat,1)
lp_mdust_concat   = np.poly1d(line_mdust_concat)

# Plotting (including straight lines)
f4, (ax10, ax11, ax12) = plt.subplots(1,3, sharey = False)
f4.tight_layout(rect=(0.0, 0.0, 1, 1))
ax10.plot(mag_params['phot_Mstar'], cig_params['phot_Mstar'] - mag_params['phot_Mstar'],
         'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax10.plot(mag_params['spec_Mstar'], cig_params['spec_Mstar'] - mag_params['spec_Mstar'],
         'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax10.plot(np.linspace(4,13,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                                         # 1:1 line
ax10.plot(np.linspace(4,13,1000), lp_mstar_phot(np.linspace(4,13,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                    # Photz sources fit
ax10.plot(np.linspace(4,13,1000), lp_mstar_spec(np.linspace(4,13,1000)), '-', color='purple',label='Specz fit', linewidth=2)                    # Specz sources fit
ax10.plot(np.linspace(4,13,1000), lp_mstar_concat(np.linspace(4,13,1000)), '-', color = 'grey', label='Total fit', linewidth=2) # All sources fit
ax10.set_xlabel('$log_{10}(M_{*,MAG})$ [$M_{\odot}$]', fontsize=20)
ax10.set_ylabel('$log_{10}(M_{*,CIG})$ - $log_{10}(M_{*,MAG})$ [$M_{\odot}$]', fontsize=20)
ax10.set_ylim(-4, 4)
ax10.set_xlim(6,13)
ax10.legend(loc='best')

ax11.plot(mag_params['phot_SFR'], cig_params['phot_SFR'] - mag_params['phot_SFR'],
         'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax11.plot(mag_params['spec_SFR'], cig_params['spec_SFR'] - mag_params['spec_SFR'], 
         'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax11.plot(np.linspace(-6,4,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                                             # 1:1 line
ax11.plot(np.linspace(-6,6,1000), lp_sSFR_phot(np.linspace(-6,6,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                     # Photz sources fit
ax11.plot(np.linspace(-6,6,1000), lp_sSFR_spec(np.linspace(-6,6,1000)), '-', color='purple', label='Specz fit', linewidth=2)                     # Specz sources fit
ax11.plot(np.linspace(-6,6,1000), lp_sSFR_concat(np.linspace(-6,6,1000)), '-', color = 'grey', label='Total fit', linewidth=2)  # All sources fit
ax11.set_xlabel('$log_{10}(SFR_{MAG})$ [$year^{(-1)}$]', fontsize=20)
ax11.set_ylabel('$log_{10}(SFR_{CIG})$ - $log_{10}(sSFR_{MAG})$ [$year^{(-1)}$]', fontsize=20)
ax11.legend(loc='best')
ax11.set_xlim(-6,4)

ax12.plot(mag_params['phot_Mdust'], cig_params['phot_Mdust'] - mag_params['phot_Mdust'],
          'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
ax12.plot(mag_params['spec_Mdust'], cig_params['spec_Mdust'] - mag_params['spec_Mdust'],
          'o', color='purple', markersize=3, alpha=0.5, label='Specz')
ax12.plot(np.linspace(0,12,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)
ax12.plot(np.linspace(0,12,1000), lp_mdust_phot(np.linspace(0,12,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)                     # Photz sources fit
ax12.plot(np.linspace(0,12,1000), lp_mdust_spec(np.linspace(0,12,1000)), '-', color='purple', label='Specz fit', linewidth=2)                     # Specz sources fit
ax12.plot(np.linspace(0,12,1000), lp_mdust_concat(np.linspace(0,12,1000)), '-', color = 'grey', label='Total fit', linewidth=2)  # All sources fit
ax12.set_xlabel('$log_{10}(M_{d,MAG})$ [$M_{\odot}$]', fontsize=20)
ax12.set_ylabel('$log_{10}(M_{d,CIG})$ - $log_{10}(M_{d,MAG})$ [$M_{\odot}$]', fontsize=20)
ax12.legend(loc='best')
ax12.set_xlim(2,12)


# Spearman rank test
spearman_mstar_photz = scipy.stats.kendalltau(mag_params['phot_Mstar'], cig_params['phot_Mstar'] - mag_params['phot_Mstar'])
spearman_sSFR_photz  = scipy.stats.kendalltau(mag_params['phot_SFR'], cig_params['phot_SFR'] - mag_params['phot_SFR'])
spearman_mdust_photz = scipy.stats.kendalltau(mag_params['phot_Mdust'], cig_params['phot_Mdust'] - mag_params['phot_Mdust'])

spearman_mstar_specz = scipy.stats.kendalltau(mag_params['spec_Mstar'], cig_params['spec_Mstar'] - mag_params['spec_Mstar'])
spearman_sSFR_specz  = scipy.stats.kendalltau(mag_params['spec_SFR'], cig_params['spec_SFR'] - mag_params['spec_SFR'])
spearman_mdust_specz = scipy.stats.kendalltau(mag_params['spec_Mdust'], cig_params['spec_Mdust'] - mag_params['spec_Mdust'])


################################################################################
### Doing the flat comparison plots (1:1 line is flat) #########################
################################################################################
# This is not the best way to plot the data 
# The best way is with the 1:1 line being flat - will make these plots here
# We also want to fit straight lines to this data to see how it compares to the 
# 1:1 line
names_phot = ['phot_Mstar', 'phot_SFR', 'phot_Mdust']
names_spec = ['spec_Mstar', 'spec_SFR', 'spec_Mdust']
# Fitting straight line to the data (three lines, the individual z and total)
# Here we want to fit m=0
# Just photz
line_phot_0 = []
lp_phot_0 = []
for i in (0,1,2):
	line_phot_00 = np.ma.polyfit(mag_params[names_phot[i]], cig_params[names_phot[i]] - mag_params[names_phot[i]],0)
	lp_phot_00   = np.poly1d(line_phot_00)
	line_phot_0.append(line_phot_00)
	lp_phot_0.append(lp_phot_00)

# Just specz
line_spec_0 = []
lp_spec_0 = []
for i in (0,1,2):
	line_spec_00 = np.ma.polyfit(mag_params[names_spec[i]], cig_params[names_spec[i]] - mag_params[names_spec[i]],0)
	lp_spec_00   = np.poly1d(line_spec_00)
	line_spec_0.append(line_spec_00)
	lp_spec_0.append(lp_spec_00)

# Both of them together (need to join data)
mag_mstar_concat_0 = np.ma.concatenate((mag_params['phot_Mstar'], mag_params['spec_Mstar']))
cig_mstar_concat_0 = np.ma.concatenate((cig_params['phot_Mstar'], cig_params['spec_Mstar']))
mag_SFR_concat_0   = np.ma.concatenate((mag_params['phot_SFR'], mag_params['spec_SFR']))
cig_SFR_concat_0   = np.ma.concatenate((cig_params['phot_SFR'], cig_params['spec_SFR']))
mag_mdust_concat_0 = np.ma.concatenate((mag_params['phot_Mdust'], mag_params['spec_Mdust']))
cig_mdust_concat_0 = np.ma.concatenate((cig_params['phot_Mdust'], cig_params['spec_Mdust']))

# The concatenated data
line_mstar_concat_0 = np.ma.polyfit(mag_mstar_concat_0, cig_mstar_concat_0 - mag_mstar_concat_0, 0)
lp_mstar_concat_0   = np.poly1d(line_mstar_concat_0)

line_SFR_concat_0 = np.ma.polyfit(mag_SFR_concat_0, cig_SFR_concat_0 - mag_SFR_concat_0, 0)
lp_SFR_concat_0   = np.poly1d(line_SFR_concat_0)

line_mdust_concat_0 = np.ma.polyfit(mag_mdust_concat_0, cig_mdust_concat_0 - mag_mdust_concat_0, 0)
lp_mdust_concat_0   = np.poly1d(line_mdust_concat_0)

# Store the data in a similar manner 
line_concat_0 = []
lp_concat_0 = []

line_concat_0.append(line_mstar_concat_0)
line_concat_0.append(line_SFR_concat_0)
line_concat_0.append(line_mdust_concat_0)

lp_concat_0.append(lp_mstar_concat_0)
lp_concat_0.append(lp_SFR_concat_0)
lp_concat_0.append(lp_mdust_concat_0)

# Now we want to make residual plots 
# In order of Mstar, SFR, Mdust
resid_photz_0 = []
resid_specz_0 = []
resid_total_0 = []
# data - model
for m in (0,1,2):
	resid_photz_0.append( (cig_params[names_phot[m]] - mag_params[names_phot[m]]) - lp_phot_0[m](mag_params[names_phot[m]]) )
	resid_specz_0.append( (cig_params[names_spec[m]] - mag_params[names_spec[m]]) - lp_spec_0[m](mag_params[names_spec[m]]) )
resid_total_0.append( (cig_mstar_concat_0 - mag_mstar_concat_0) - lp_concat_0[0](mag_mstar_concat_0) )
resid_total_0.append( (cig_SFR_concat_0 - mag_SFR_concat_0) - lp_concat_0[1](mag_SFR_concat_0) )
resid_total_0.append( (cig_mdust_concat_0 - mag_mdust_concat_0) - lp_concat_0[2](mag_mdust_concat_0) )

"""
# Fit a straight line to the residuals
residline_photz_0 = []
residlp_photz_0   = []
residline_specz_0 = []
residlp_specz_0   = []
residline_total_0 = []
residlp_total_0   = []
for m in (0,1,2):
	# Photz
	lineresid_p = np.polyfit(mag_params[names_phot[m]], resid_photz_0[m], 1)
	residline_photz_0.append(lineresid_p)
	residlp_photz_0.append(np.poly1d(lineresid_p))
	# Specz
	lineresid_s = np.polyfit(mag_params[names_spec[m]], resid_specz_0[m], 1)
	residline_specz_0.append(lineresid_s)
	residlp_specz_0.append(np.poly1d(lineresid_s))
# Total
# Mstar
lineresid_t = np.polyfit(mag_mstar_concat_0, resid_total_0[0], 1)
residline_total_0.append(lineresid_t)
residlp_total_0.append(np.poly1d(lineresid_t))
# SFR
lineresid_t = np.polyfit(mag_SFR_concat_0, resid_total_0[1], 1)
residline_total_0.append(lineresid_t)
residlp_total_0.append(np.poly1d(lineresid_t))
# Mdust
lineresid_t = np.polyfit(mag_mdust_concat_0, resid_total_0[2], 1)
residline_total_0.append(lineresid_t)
residlp_total_0.append(np.poly1d(lineresid_t))	
"""	

# We want to calculate the dispersion about the residuals for each of the parameters
# and for the three sets of data that we are considering
# 1) all the data 
# 2) photz data
# 3) specz data
def dispersion(residuals):
	# The dispersion is essentially the root mean square
	# You square the residuals - sum them - divide by no. points - square root
	# First we need to check if the data array is masked
	if np.ma.is_masked(residuals) == True:
		# Extract only the unmasked values
		residuals = residuals[~residuals.mask].data
	# Calculations
	resid_sq = residuals**2
	sumresid = np.sum(resid_sq)
	divide = sumresid / np.float(len(residuals))
	dispers = np.sqrt(divide)
	return float("{0:.2f}".format(dispers))

# Calculating the dispersion
# Order is Mstar, SFR, Mdust
disper_total_0 = []
disper_photz_0 = []
disper_specz_0 = []
for i in (0,1,2):
	disper_total_0.append(dispersion(resid_total_0[i])) # total data
	disper_photz_0.append(dispersion(resid_photz_0[i])) # photz data
	disper_specz_0.append(dispersion(resid_specz_0[i])) # specz data

axes_x_labels = ['$log_{10}(M_{*,MAG})$ [$M_{\odot}$]', '$log_{10}(SFR_{MAG})$ [$year^{(-1)}$]', '$log_{10}(M_{d,MAG})$ [$M_{\odot}$]']
axes_y_labels = ['$log_{10}(M_{*,CIG})$ - $log_{10}(M_{*,MAG})$ [$M_{\odot}$]', '$log_{10}(SFR_{CIG})$ - $log_{10}(sSFR_{MAG})$ [$year^{(-1)}$]',
                 '$log_{10}(M_{d,CIG})$ - $log_{10}(M_{d,MAG})$ [$M_{\odot}$]']

#plt.close('all')

# Plotting (including straight lines)
f5, ((ax13, ax14, ax15), (ax16, ax17, ax18), (ax19, ax20, ax21), (ax22, ax23, ax24)) = plt.subplots(4,3, sharey = False, gridspec_kw = {'height_ratios': [4,1,1,1]})
f5.subplots_adjust(hspace=0, left = 0.05, right=0.95, top=0.95)

for j in (0,1,2):
	if j == 0:
		axx  = ax13 # main plotting axis
		axr1 = ax16 # total residual (1st residual)
		axr2 = ax19 # photz residual (2nd residual)
		axr3 = ax22 # specz residual (3rd residual)
		xy_text = (0.75,0.85) # location of annotation
		datarange = np.linspace(4,13,1000)
		concat_dat = mag_mstar_concat_0
	elif j == 1:
		axx = ax14  # main plotting axis
		axr1 = ax17 # total residual (1st residual)
		axr2 = ax20 # photz residual (2nd residual)
		axr3 = ax23 # specz residual (3rd residual)
		xy_text = (0.75,0.85) # location of annotation
		datarange = np.linspace(-6,6,1000)
		concat_dat = mag_SFR_concat_0
	elif j == 2:
		axx = ax15  # main plotting axis
		axr1 = ax18 # total residual (1st residual)
		axr2 = ax21 # photz residual (2nd residual)
		axr3 = ax24 # specz residual (3rd residual)
		xy_text = (0.75,0.85) # location annotation
		datarange = np.linspace(0,12,1000)
		concat_dat = mag_mdust_concat_0

	print('Now plotting: '),
	print(names_phot[j])
	print(' ')

	axx.plot(mag_params[names_phot[j]], cig_params[names_phot[j]] - mag_params[names_phot[j]],
		     'o', color='darkorange', markersize=3, alpha=0.5, label='Photz')
	axx.plot(mag_params[names_spec[j]], cig_params[names_spec[j]] - mag_params[names_spec[j]],
		     'o', color='purple', markersize=3, alpha=0.5, label='Specz')
	axx.plot(datarange, np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                              # 1:1 line
	axx.plot(datarange, lp_phot_0[j](datarange), '-', color='darkorange', label='Photz fit', linewidth=2)    # Photz sources fit
	axx.plot(datarange, lp_spec_0[j](datarange), '-', color='purple',label='Specz fit', linewidth=2)         # Specz sources fit
	axx.plot(datarange, lp_concat_0[j](datarange), '-', color = 'grey', label='Total fit', linewidth=2)      # All sources fit
	axr1.plot(concat_dat, resid_total_0[j], 'o', color='grey', markersize=3, alpha=0.5)                      # Resids total
	axr1.plot(datarange, np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                             # 1:1 line
	axr2.plot(mag_params[names_phot[j]], resid_photz_0[j], 'o', color='darkorange', markersize=3, alpha=0.5) # Resids photz
	axr2.plot(datarange, np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                             # 1:1 line
	axr3.plot(mag_params[names_spec[j]], resid_specz_0[j], 'o', color='purple', markersize=3, alpha=0.5)     # Resids photz
	axr3.plot(datarange, np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                             # 1:1 line
	# Annotating the dispersion
	axr1.annotate('$\Delta_{Total}$'+' = '+str(disper_total_0[j]), xy = xy_text, xycoords='axes fraction', fontsize=14)
	axr2.annotate('$\Delta_{Photz}$'+' = '+str(disper_photz_0[j]), xy = xy_text, xycoords='axes fraction', fontsize=14)
	axr3.annotate('$\Delta_{Specz}$'+' = '+str(disper_specz_0[j]), xy = xy_text, xycoords='axes fraction', fontsize=14)
	# Labelling
	axr3.set_xlabel(axes_x_labels[j], fontsize=20)
	axx.set_ylabel(axes_y_labels[j], fontsize=20)
	axr1.set_ylabel('$Residual$', fontsize=14)
	axr2.set_ylabel('$Residual$', fontsize=14)
	axr3.set_ylabel('$Residual$', fontsize=14)
	axx.legend(loc='best', numpoints=1, prop={'size':12})
	axx.axes.get_xaxis().set_visible(False)
	axr1.axes.get_xaxis().set_visible(False)
	axr2.axes.get_xaxis().set_visible(False)
	plt.setp(axr1.get_yticklabels()[0], visible=False)
	plt.setp(axr1.get_yticklabels()[-1], visible=False)
	plt.setp(axr2.get_yticklabels()[0], visible=False)
	plt.setp(axr2.get_yticklabels()[-1], visible=False)
	plt.setp(axr3.get_yticklabels()[0], visible=False)
	plt.setp(axr3.get_yticklabels()[-1], visible=False)

ax13.set_ylim(-3, 3)
ax13.set_xlim(6,13)
ax16.set_xlim(6,13)
ax19.set_xlim(6,13)
ax22.set_xlim(6,13)
ax14.set_xlim(-6,4)
ax17.set_xlim(-6,4)
ax20.set_xlim(-6,4)
ax23.set_xlim(-6,4)
ax15.set_xlim(2,12)
ax18.set_xlim(2,12)
ax21.set_xlim(2,12)
ax24.set_xlim(2,12)


################################################################################
### Density plots to try and better see the distribution of points #############
################################################################################
# Going to try and make a density plot of the dust mass, a subplot of three panels
# which is the photometric redshifts, the spectral redshifts and then in total.
# Can then plot the lines over the top to see if they are sensible

# Create a function which will return the x,y,z values for plotting
# This allows us to create a density plot to illustrate which parts of colour
# space are well covered by the models
def colour_density(xdata,ydata):
	
	#This function will return the z value which is the colour scale value when 
	#creating density plots.
	
	# Calculating the point density (this is used in the colour scheme)
	xy = np.vstack([xdata,ydata])
	z = gaussian_kde(xy)(xy)
	# Sort the points by density so the most dense is plotted last
	idx = z.argsort()
	x, y, z = xdata[idx], ydata[idx], z[idx]
	return x,y,z

# Plotting
# Just photz data
fig = plt.figure()
fig.subplots_adjust(left=0.05, right= 0.98, bottom=0.06, top=0.95, wspace=0.1, hspace=0.05)
ax30 = fig.add_subplot(1,3,1)
x,y,z = colour_density(mag_params['phot_Mdust'], cig_params['phot_Mdust'] - mag_params['phot_Mdust'])
ax30.scatter(x,y,c=z, cmap='inferno', lw=0)
ax30.plot(np.linspace(2,13,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                       # 1:1 line
ax30.plot(np.linspace(2,13,1000), lp_mdust_phot(np.linspace(2,13,1000)), '-', color='darkorange', label='Photz fit', linewidth=2)  # Photz sources fit
ax30.set_xlabel('$log_{10}(M_{g,Sco16})$ [$10^{10}M_{\odot}$]', fontsize=20)
ax30.set_ylabel('$M_{g,CIG}$ - $M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
ax30.set_ylim(-4,6)
ax30.set_xlim(2,12)
ax30.set_title('Photo-z')

# Just specz data
ax31 = fig.add_subplot(1,3,2)
x,y,z = colour_density(mag_params['spec_Mdust'], cig_params['spec_Mdust'] - mag_params['spec_Mdust'])
ax31.scatter(x,y,c=z, cmap='inferno', lw=0)
ax31.plot(np.linspace(2,13,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                      # 1:1 line
ax31.plot(np.linspace(2,13,1000), lp_mdust_spec(np.linspace(2,13,1000)), '-', color='purple', label='Specz fit', linewidth=2) # Specz sources fit
ax31.set_xlabel('$log_{10}(M_{g,Sco16})$ [$10^{10}M_{\odot}$]', fontsize=20)
ax31.set_ylim(-4,6)
ax31.set_xlim(2,12)
ax31.set_title('Spec-z')

# Total data
ax32 = fig.add_subplot(1,3,3)
x,y,z = colour_density(mag_mdust_concat, cig_mdust_concat - mag_mdust_concat)
ax32.scatter(x,y,c=z, cmap='inferno', lw=0)
ax32.plot(np.linspace(2,13,1000), np.linspace(0,0,1000), 'k--', label='1:1', linewidth=2)                      # 1:1 line
ax32.plot(np.linspace(2,13,1000), lp_mdust_concat(np.linspace(2,13,1000)), '-', color = 'grey', label='Total fit', linewidth=2) # All sources fit
ax32.set_xlabel('$log_{10}(M_{g,Sco16})$ [$10^{10}M_{\odot}$]', fontsize=20)
ax32.set_ylim(-4,6)
ax32.set_xlim(2,12)
ax32.set_title('Total')


plt.show('all')




