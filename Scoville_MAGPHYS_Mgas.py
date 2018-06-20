"""
This code will plot Mgas against Mgas for 145 galaxies from the Scoville 2016 
paper. The our Mgas comes from Mdust from MAGPHYS and we assume a GDR of 100, 
which is the average for the Milky Way but not necessarily true for these galaxies
but it is a good start. 

We then plot the 1:1 line and we also fit a straight line to the data. 

Errors for magphys data will be the 16th and 84th percentiles.

Errors for Scoville 2016 are as quoted. If the gas mass quoted is an upper limit,
we plot at a data point with a error down to zero(?)
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

from mpl_toolkits.axes_grid import AxesGrid, make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

from itertools import compress

from matplotlib import gridspec

plt.close('all')
################################################################################
### Importing and extracting data ##############################################
################################################################################
# General path
gen_path = '/home/prongs/c1124553/PhD_Year2/Cosmos_DMF/Cross_matched_data/Minimized_tables/'

# Importing data from MAGPHYS and Scoville2016
lowz_M  = pd.read_csv(gen_path+'LAMBDAR_MAGPHYS_Lowz_Sco_1arcsec.csv')
midz_M  = pd.read_csv(gen_path+'LAMBDAR_MAGPHYS_Midz_Sco_1arcsec.csv')
highz_M = pd.read_csv(gen_path+'LAMBDAR_MAGPHYS_Highz_Sco_1arcsec.csv')
# Importing data from CIGALE and Scoville2016
lowz_C  = pd.read_csv(gen_path+'CIGALE_photz_Lowz_Sco_1arcsec.csv')
midz_C  = pd.read_csv(gen_path+'CIGALE_photz_Midz_Sco_1arcsec.csv')
highz_C = pd.read_csv(gen_path+'CIGALE_photz_Highz_Sco_1arcsec.csv')


# Extract the data and errors we are interested in
# First the Scoville data
colnames = ['Mg_lowz','Mg_midz','Mg_highz','Mg_lowzerr','Mg_midzerr','Mg_highzerr']
dd_sco = {}
for h in ['_M', '_C']: # We need to loop for the CIGALE data and MAGPHYS
	if h == '_M': # MAGPHYS matches
		dd_sco[colnames[0]+h] = lowz_M['Mmol']
		dd_sco[colnames[1]+h] = midz_M['Mmol']
		dd_sco[colnames[2]+h] = highz_M['Mmol']
		dd_sco[colnames[3]+h] = lowz_M['Mmolerr']
		dd_sco[colnames[4]+h] = midz_M['Mmolerr']
		dd_sco[colnames[5]+h] = highz_M['Mmolerr']
	elif h == '_C': # CIGALE matches
		dd_sco[colnames[0]+h] = lowz_C['Mmol']
		dd_sco[colnames[1]+h] = midz_C['Mmol']
		dd_sco[colnames[2]+h] = highz_C['Mmol']
		dd_sco[colnames[3]+h] = lowz_C['Mmolerr']
		dd_sco[colnames[4]+h] = midz_C['Mmolerr']
		dd_sco[colnames[5]+h] = highz_C['Mmolerr']

# Next the MAGPHYS data
# This is dust mass for now
colnames2 = ['Md_lowz','Md_midz','Md_highz']
dd_magphys = {}
dd_magphys[colnames2[0]]      = lowz_M['mass_dust_percentile50']
dd_magphys[colnames2[0]+'16'] = lowz_M['mass_dust_percentile16']
dd_magphys[colnames2[0]+'84'] = lowz_M['mass_dust_percentile84']
dd_magphys[colnames2[1]]      = midz_M['mass_dust_percentile50']
dd_magphys[colnames2[1]+'16'] = midz_M['mass_dust_percentile16']
dd_magphys[colnames2[1]+'84'] = midz_M['mass_dust_percentile84']
dd_magphys[colnames2[2]]      = highz_M['mass_dust_percentile50']
dd_magphys[colnames2[2]+'16'] = highz_M['mass_dust_percentile16']
dd_magphys[colnames2[2]+'84'] = highz_M['mass_dust_percentile84']

# Now the CIGALE data
# Again, we just want the dust mass for now
dd_cigale = {}
dd_cigale[colnames2[0]]        = lowz_C['UVoptIR_bayes.dust.mass']
dd_cigale[colnames2[0]+'_err'] = lowz_C['UVoptIR_bayes.dust.mass_err']
dd_cigale[colnames2[1]]        = midz_C['UVoptIR_bayes.dust.mass']
dd_cigale[colnames2[1]+'_err'] = midz_C['UVoptIR_bayes.dust.mass_err']
dd_cigale[colnames2[2]]        = highz_C['UVoptIR_bayes.dust.mass']
dd_cigale[colnames2[2]+'_err'] = highz_C['UVoptIR_bayes.dust.mass_err']

# Units need to be the same
# Scoville in units of (10^10)Msun
# Magphys in units of log(Md)Msun
# Cigale in units of kg
# Let's convert the magphys stuff into units of Msun for all
for i in [0,1,2]:
	dd_magphys[colnames2[i]]      = pow(dd_magphys[colnames2[i]],10)
	dd_magphys[colnames2[i]+'16'] = pow(dd_magphys[colnames2[i]+'16'],10)
	dd_magphys[colnames2[i]+'84'] = pow(dd_magphys[colnames2[i]+'84'],10)
# Now we need to convert the CIGALE dust masses into units of (10^10)Msun
masssun = 1.989E30 # mass of sun in kg
for j in [0,1,2]:
	dd_cigale[colnames2[j]]        = (dd_cigale[colnames2[j]] / masssun)
	dd_cigale[colnames2[j]+'_err'] = (dd_cigale[colnames2[j]+'_err'] / masssun)
# And we need to convert the Scoville masses
for h in ['_M', '_C']: # We need to loop for the CIGALE data and MAGPHYS
	dd_sco[colnames[0]+h] = dd_sco[colnames[0]+h]*(1E10)
	dd_sco[colnames[1]+h] = dd_sco[colnames[1]+h]*(1E10)
	dd_sco[colnames[2]+h] = dd_sco[colnames[2]+h]*(1E10)
	dd_sco[colnames[3]+h] = dd_sco[colnames[3]+h]*(1E10)
	dd_sco[colnames[4]+h] = dd_sco[colnames[4]+h]*(1E10)
	dd_sco[colnames[5]+h] = dd_sco[colnames[5]+h]*(1E10)

# But this is the dust mass, we want the gas mass, which we can now calculate
# because we have everything in the same units
# We are assuming a gas-to-dust ratio of 100, so multiply through by 100
# Need to multiply errors by 100 also
# Make use of colnames
for j in [0,1,2]:
	dd_magphys[colnames[j]]      = dd_magphys[colnames2[j]]*100
	dd_magphys[colnames[j]+'16'] = dd_magphys[colnames2[j]+'16']*100
	dd_magphys[colnames[j]+'84'] = dd_magphys[colnames2[j]+'84']*100
	dd_cigale[colnames[j]] = dd_cigale[colnames2[j]]*100
	dd_cigale[colnames[j]+'_err'] = dd_cigale[colnames2[j]+'_err']*100

"""
################################################################################
### Plotting data ##############################################################
################################################################################
# Quick plot of MAGPHYS data only 
labelnames = ['$z \sim 1.15$','$z \sim 2.2$','$z \sim 4.4$']
plt.figure(1)
colpoint = ['blue', 'green', 'red']
for i in [0,1,2]:
	plt.plot(dd_sco[colnames[i]+'_M'], dd_magphys[colnames[i]], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
plt.plot(np.linspace(0,100,1000), np.linspace(0,100,1000), '-', color='k')
plt.legend(loc='best')
plt.xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
plt.ylabel('$M_{g,MAG}$ [$10^{10}M_{\odot}$]', fontsize=20)
	
# Plot subplots = LHS is no errorbars, RHS is with errorbars 
f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
f.tight_layout(pad=1.2)
# LHS, no errorbars
for i in [0,1,2]:
	ax1.plot(dd_sco[colnames[i]+'_M'], dd_magphys[colnames[i]], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
ax1.plot(np.linspace(0,100,1000), np.linspace(0,100,1000), '-', color='k', label='1:1')
ax1.legend(loc='best')
ax1.set_xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
ax1.set_ylabel('$M_{g,MAG}$ [$10^{10}M_{\odot}$]', fontsize=20)
# RHS, with errorbars
# We have to do some fiddling with the Scoville error bars to illustrate upper 
# limits
for i in [0,1,2]:
	# Extract the data for plotting
	xdata = dd_sco[colnames[i]+'_M']
	xdataerrs = dd_sco[colnames[i+3]+'_M']
	ydata = dd_magphys[colnames[i]]
	ydatauperr = dd_magphys[colnames[i]+'84'] - dd_magphys[colnames[i]]
	ydatalowerr = dd_magphys[colnames[i]] - dd_magphys[colnames[i]+'16']
	# Now we need to loop over each point for plotting and if for the Scoville
	# data it is only an upper limit, plot in a different symbol e.g. triangle
	for j in range(0, len(xdata)):
		# If the Scoville error is not -99.0, plot as usual
		if (xdataerrs[j] != -99.0):
			ax2.errorbar(xdata[j], ydata[j], xerr=xdataerrs[j], yerr=np.asarray([[ydatalowerr[j], ydatauperr[j]]]).transpose(),
                         fmt='o', markersize=5, color=colpoint[i], linestyle='None')
		# If the Scoville error is -99.0, plot as upper limit
		elif (xdataerrs[j] == -99.0):
			ax2.errorbar(xdata[j], ydata[j], xerr=np.asarray([[xdata[j], 0]]).transpose(), yerr=np.asarray([[ydatalowerr[j], ydatauperr[j]]]).transpose(), 
                         fmt='<', markersize=5, color=colpoint[i], linestyle='None')
ax2.plot(np.linspace(0,100,1000), np.linspace(0,100,1000), '-', color='k', label='1:1')
ax2.set_xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)


################################################################################
### Fitting a straight line to the data (no error consideration) ###############
################################################################################
# MAGPHYS data only
# Concatenate the data
xdataline = np.concatenate((dd_sco[colnames[0]+'_M'].values, dd_sco[colnames[1]+'_M'].values, dd_sco[colnames[2]+'_M'].values))
ydataline = np.concatenate((dd_magphys[colnames[0]].values, dd_magphys[colnames[1]].values, dd_magphys[colnames[2]].values))
# Fit the line
line = np.polyfit(xdataline, ydataline, 1)
lp = np.poly1d(line)

# Quick plot
plt.figure(3)
colpoint = ['blue', 'green', 'red']
for i in [0,1,2]:
	plt.plot(dd_sco[colnames[i]+'_M'], dd_magphys[colnames[i]], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
plt.plot(np.linspace(0,100,1000), np.linspace(0,100,1000), '-', color='k', label='1:1')
plt.plot(np.linspace(0,100,1000), lp(np.linspace(0,100,1000)), '-', color='m', label='BF')
plt.legend(loc="best")
plt.xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
plt.ylabel('$M_{g,MAG}$ [$10^{10}M_{\odot}$]', fontsize=20)

# Fit straight line to data - errors in x and y?


################################################################################
### Fit to the different redshifts (no error consideration) ####################
################################################################################
# MAGPHYS data only
# Looping
lineparams = []
for i in (0,1,2):
	# Extract the data
	xdataline1 = dd_sco[colnames[i]+'_M'].values
	ydataline1 = dd_magphys[colnames[i]].values
	# Fit the line
	line1 = np.polyfit(xdataline1, ydataline1, 1)
	lp1 = np.poly1d(line1)
	# Save for later
	lineparams.append(lp1)

# Plot these 
plt.figure(4)
colpoint  = ['blue', 'green', 'red']
linesty = ['--', '-.', ':'] 
for i in [0,1,2]:
	linedat = lineparams[i]
	plt.plot(dd_sco[colnames[i]+'_M'], dd_magphys[colnames[i]], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
	plt.plot(np.linspace(0,100,1000), linedat(np.linspace(0,100,1000)), linestyle=linesty[i], color=colpoint[i], label=labelnames[i]+' BF')
plt.plot(np.linspace(0,100,1000), np.linspace(0,100,1000), '-', color='k', label='1:1')
plt.plot(np.linspace(0,100,1000), lp(np.linspace(0,100,1000)), '-', color='m', label='BF')
plt.legend(loc="best")
plt.xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
plt.ylabel('$M_{g,MAG}$ [$10^{10}M_{\odot}$]', fontsize=20)

plt.close('all')
"""

################################################################################
### Log-space plots to examine scatter of points ###############################
################################################################################
# Want to log10 all of the gas masses for plotting
# Note we are not considering errors at this stage
# MAGPHYS and CIGALE data
for i in (0,1,2):
	dd_sco[colnames[i]+'_M_log'] = np.log10(dd_sco[colnames[i]+'_M']) # Scoville gas mass - MAGPHYS matches
	dd_sco[colnames[i]+'_C_log'] = np.log10(dd_sco[colnames[i]+'_C']) # Scoville gas mass - CIGALE matches
	dd_magphys[colnames[i]+'-sco_log'] = np.log10(dd_magphys[colnames[i]]) - np.log10(dd_sco[colnames[i]+'_M']) # Magphys gas mass - Scoville gas mass
	dd_cigale[colnames[i]+'-sco_log'] = np.log10(dd_cigale[colnames[i]]) - np.log10(dd_sco[colnames[i]+'_C'])   # Cigale gas mass - Scoville gas mass

# Plotting definitions
error_filt = -99.0E10
labelnames = ['$z \sim 1.15$','$z \sim 2.2$','$z \sim 4.4$']
colpoint = ['blue', 'green', 'red']
	
# Now we can make the basic plot - no straight line fitting, no residuals yet
# We want to plot the data as a different symbol if it is an upper limit
# Plotting the MAGPHYS data
# Quick plot of MAGPHYS data only 
plt.figure(5)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2)
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,MAG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)


# Now we can make the basic plot - no straight line fitting, no residuals yet
# We want to plot the data as a different symbol if it is an upper limit
# Plotting the CIGALE data
plt.figure(6)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=0): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==0):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==0):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2)
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,CIG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)

##
####
##

"""
# Not logging y-axis here
# Note we are not considering errors at this stage
# MAGPHYS and CIGALE data
for i in (0,1,2):
	dd_magphys[colnames[i]+'-sco'] = dd_magphys[colnames[i]] - dd_sco[colnames[i]+'_M'] # Magphys gas mass - Scoville gas mass
	dd_cigale[colnames[i]+'-sco'] = dd_cigale[colnames[i]] - dd_sco[colnames[i]+'_C']   # Cigale gas mass - Scoville gas mass
	
# Now we can make the basic plot - no straight line fitting, no residuals yet
# We want to plot the data as a different symbol if it is an upper limit
# Plotting the MAGPHYS data
plt.figure(7)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != -99.0) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != -99.0) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == -99.0): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(0,2.0,1000), np.zeros(1000), '--', color='k', linewidth=2)
plt.legend(loc="best")
#plt.ylim(-30,70)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$10^{10}M_{\odot}$]', fontsize=20)
plt.ylabel('$M_{g,MAG}$ - $M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)


# Now we can make the basic plot - no straight line fitting, no residuals yet
# We want to plot the data as a different symbol if it is an upper limit
# Plotting the CIGALE data
plt.figure(8)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != -99.0) and (j!=0): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != -99.0) and (j==0):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == -99.0) and (j==0):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == -99.0): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(0,2.0,1000), np.zeros(1000), '--', color='k', linewidth=2)
plt.legend(loc="best")
#plt.ylim(-30,70)
plt.xlabel('$M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
plt.ylabel('$M_{g,CIG}$ - $M_{g,Sco16}$ [$10^{10}M_{\odot}$]', fontsize=20)
"""

plt.close('all')
################################################################################
### Fitting straight lines to log plots ########################################
################################################################################
# So we clearly do not have the relationship that we wanted, there is a clear
# trend. So now we fit these data points with a straight line (again, ignoring
# errors for now)
# Concatenate the data
# MAGPHYS
xlogline_M = np.concatenate((dd_sco[colnames[0]+'_M_log'].values, dd_sco[colnames[1]+'_M_log'].values, dd_sco[colnames[2]+'_M_log'].values))
ylogline_M = np.concatenate((dd_magphys[colnames[0]+'-sco_log'].values, dd_magphys[colnames[1]+'-sco_log'].values, dd_magphys[colnames[2]+'-sco_log'].values))
# Fit the line MAGPHYS
logline_M = np.polyfit(xlogline_M, ylogline_M, 1)
loglp_M = np.poly1d(logline_M)
# CIGALE
xlogline_C = np.concatenate((dd_sco[colnames[0]+'_C_log'].values, dd_sco[colnames[1]+'_C_log'].values, dd_sco[colnames[2]+'_C_log'].values))
ylogline_C = np.concatenate((dd_cigale[colnames[0]+'-sco_log'].values, dd_cigale[colnames[1]+'-sco_log'].values, dd_cigale[colnames[2]+'-sco_log'].values))
# Fit the line CIGALE
logline_C = np.polyfit(xlogline_C, ylogline_C, 1)
loglp_C = np.poly1d(logline_C)

# Fit a line with a gradient m=0 so we just find the intercept
# Fit to the MAGPHYS data
logline_M0 = np.polyfit(xlogline_M, ylogline_M, 0)
loglp_M0 = np.poly1d(logline_M0)
# Fit to the CIGALE data
logline_C0 = np.polyfit(xlogline_C, ylogline_C, 0)
loglp_C0 = np.poly1d(logline_C0)


# Also fit the data excluding the upper limits on the Scoville data
# Create a mask, then refit the line
# Mask where the unlogged xdata-error is -99.0 
# MAGPHYS
dd_ma_magsco = {}
for i in (0,1,2):
	xdataerr = dd_sco[colnames[i+3]+'_M']
	maskk = xdataerr < -90.0
	# Now we need to mask the relevant data 
	xdata_ma = np.ma.array(dd_sco[colnames[i]+'_M_log'], mask=maskk)
	ydata_ma = np.ma.array(dd_magphys[colnames[i]+'-sco_log'], mask=maskk)
	# Add into a new dictionary for concatentation later
	dd_ma_magsco['sco_'+colnames[i]+'_malog'] = xdata_ma
	dd_ma_magsco['magsco_'+colnames[i]+'_malog'] = ydata_ma
# Now we can concatenate the data together (also need to concatenate the masks)
xlogline_maM = np.ma.concatenate((dd_ma_magsco['sco_'+colnames[0]+'_malog'],
                                 dd_ma_magsco['sco_'+colnames[1]+'_malog'],
                                 dd_ma_magsco['sco_'+colnames[2]+'_malog']))
ylogline_maM = np.ma.concatenate((dd_ma_magsco['magsco_'+colnames[0]+'_malog'],
                                 dd_ma_magsco['magsco_'+colnames[1]+'_malog'],
                                 dd_ma_magsco['magsco_'+colnames[2]+'_malog']))
# Fit the line as normal
logline_maM = np.ma.polyfit(xlogline_maM, ylogline_maM, 1)
loglp_maM = np.poly1d(logline_maM)
# Fit the line with a gradient m=0
logline_maM0 = np.ma.polyfit(xlogline_maM, ylogline_maM, 0)
loglp_maM0 = np.poly1d(logline_maM0)

# Also fit the data excluding the upper limits on the Scoville data
# Create a mask, then refit the line
# Mask where the unlogged xdata-error is -99.0 
# CIGALE
dd_ma_cigsco = {}
for i in (0,1,2):
	xdataerr = dd_sco[colnames[i+3]+'_C']
	maskk = xdataerr < -90.0
	# Now we need to mask the relevant data 
	xdata_ma = np.ma.array(dd_sco[colnames[i]+'_C_log'], mask=maskk)
	ydata_ma = np.ma.array(dd_cigale[colnames[i]+'-sco_log'], mask=maskk)
	# Add into a new dictionary for concatentation later
	dd_ma_cigsco['sco_'+colnames[i]+'_malog'] = xdata_ma
	dd_ma_cigsco['cigsco_'+colnames[i]+'_malog'] = ydata_ma
# Now we can concatenate the data together (also need to concatenate the masks)
xlogline_maC = np.ma.concatenate((dd_ma_cigsco['sco_'+colnames[0]+'_malog'],
                                 dd_ma_cigsco['sco_'+colnames[1]+'_malog'],
                                 dd_ma_cigsco['sco_'+colnames[2]+'_malog']))
ylogline_maC = np.ma.concatenate((dd_ma_cigsco['cigsco_'+colnames[0]+'_malog'],
                                 dd_ma_cigsco['cigsco_'+colnames[1]+'_malog'],
                                 dd_ma_cigsco['cigsco_'+colnames[2]+'_malog']))
# Fit the line as normal
logline_maC = np.ma.polyfit(xlogline_maC, ylogline_maC, 1)
loglp_maC = np.poly1d(logline_maC)
# Fit the line with a gradient m=0
logline_maC0 = np.ma.polyfit(xlogline_maC, ylogline_maC, 0)
loglp_maC0 = np.poly1d(logline_maC0)

####################
##### Residual calcs
####################
# We want to calculate the residual in y for the data points away from the m=0 lines
# Model - data
resid_M0 = ylogline_M - loglp_M0(xlogline_M)
resid_C0 = ylogline_C - loglp_C0(xlogline_C)
resid_maM0 = ylogline_maM - loglp_maM0(xlogline_maM)
resid_maC0 = ylogline_maC - loglp_maC0(xlogline_maC)

####################
##### Normal fitting
####################
# Now we have fitted the data, we want to plot the data with the lines - there
# are two lines fitted, one with and one without the upper limit data
# Again, not plotting error bars yet
# MAGPHYS
plt.figure(7)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
plt.plot(np.linspace(10,12,1000), loglp_M(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
plt.plot(np.linspace(10,12,1000), loglp_maM(np.linspace(10,12,1000)), '-', color='purple', linewidth=2, label='$W$/$o$ $upper$ $limits$') # data fit
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,MAG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)

# Now we have fitted the data, we want to plot the data with the lines - there
# are two lines fitted, one with and one without the upper limit data
# Again, not plotting error bars yet
# CIGALE
plt.figure(8)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
plt.plot(np.linspace(10,12,1000), loglp_C(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
plt.plot(np.linspace(10,12,1000), loglp_maC(np.linspace(10,12,1000)), '-', color='purple', linewidth=2, label='$W$/$o$ $upper$ $limits$') # data fit
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,CIG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)

#################################################
#### Fixed gradient fitting - with residual plots
#################################################
##
### MAGPHYS
##
fig9 = plt.figure(9)
gs9 = gridspec.GridSpec(3,1, height_ratios=[3,1,1], hspace=0.0) # preparing for resid plot
ax90 = plt.subplot(gs9[0]) # creating the axes for plotting
ax91 = plt.subplot(gs9[1])
ax92 = plt.subplot(gs9[2])
# Main figure plotting
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			ax90.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			ax90.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			ax90.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
ax90.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax90.plot(np.linspace(10,12,1000), loglp_M0(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
ax90.plot(np.linspace(10,12,1000), loglp_maM0(np.linspace(10,12,1000)), '-', color='purple', linewidth=2, label='$W$/$o$ $upper$ $limits$') # data fit

# Residual figure plotting
# Remember to plot the triangles for upper limits
xdataerr2 = np.concatenate((dd_sco[colnames[0+3]+'_M'], dd_sco[colnames[1+3]+'_M'], dd_sco[colnames[2+3]+'_M']))
for j in range(0, len(xdataerr2)):
	if (xdataerr2[j] != error_filt) and (j!=3):
		ax91.plot(xlogline_M[j], resid_M0[j], color='orange', marker='o', markersize=5, linestyle='None') # residuals w/ upper limits
	elif (xdataerr2[j] != error_filt) and (j==3):
		ax91.plot(xlogline_M[j], resid_M0[j], color='orange', marker='o', markersize=5, linestyle='None', label='$W$/ $upper$ $limits$') # residuals w/ upper limits
	elif (xdataerr2[j] == error_filt): 
		ax91.plot(xlogline_M[j], resid_M0[j], color='orange', marker='<', markersize=5, linestyle='None') # residuals w/ upper limits

# No upper limits on this plot, just need to sort out legend
ax92.plot(xlogline_maM[0], resid_maM0[0], color='purple', marker='o', markersize=5, linestyle='None', label='$W$/$o$ $upper$ $limits$') # residuals w/o upper limits
ax92.plot(xlogline_maM, resid_maM0, color='purple', marker='o', markersize=5, linestyle='None') # residuals w/o upper limits

ax91.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax92.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline

# Labelling
ax90.set_title('M=0')
ax90.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax91.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.setp(ax91.get_yticklabels()[0], visible=False)
plt.setp(ax91.get_yticklabels()[-1], visible=False)
plt.setp(ax92.get_yticklabels()[0], visible=False)
plt.setp(ax92.get_yticklabels()[-1], visible=False)
ax92.set_xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax90.set_ylabel('$log_{10}(M_{g,MAG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax91.set_ylabel('$Resid$', fontsize=16)
ax92.set_ylabel('$Resid$', fontsize=16)
ax90.legend(loc="best", numpoints=1)
ax91.legend(loc="best", numpoints=1)
ax92.legend(loc="best", numpoints=1)
ax90.set_xlim(10,12)
ax90.set_ylim(-1.0,1.4)
ax91.set_ylim(-0.8,1.0)
ax92.set_ylim(-0.8,1.0)

##
### CIGALE
##
fig10 = plt.figure(10)
gs10 = gridspec.GridSpec(3,1, height_ratios=[3,1,1], hspace=0.0) # preparing for resid plot
ax100 = plt.subplot(gs10[0]) # creating the axes for plotting
ax101 = plt.subplot(gs10[1])
ax102 = plt.subplot(gs10[2])
# Main figure plotting
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			ax100.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			ax100.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==3):
			ax100.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			ax100.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
ax100.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax100.plot(np.linspace(10,12,1000), loglp_C0(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
ax100.plot(np.linspace(10,12,1000), loglp_maC0(np.linspace(10,12,1000)), '-', color='purple', linewidth=2, label='$W$/$o$ $upper$ $limits$') # data fit

# Residual figure plotting
# Remember to plot the triangles for upper limits
xdataerr3 = np.concatenate((dd_sco[colnames[0+3]+'_C'], dd_sco[colnames[1+3]+'_C'], dd_sco[colnames[2+3]+'_C']))
for j in range(0, len(xdataerr3)):
	if (xdataerr3[j] != error_filt) and (j!=3):
		ax101.plot(xlogline_C[j], resid_C0[j], color='orange', marker='o', markersize=5, linestyle='None') # residuals w/ upper limits
	elif (xdataerr3[j] != error_filt) and (j==3):
		ax101.plot(xlogline_C[j], resid_C0[j], color='orange', marker='o', markersize=5, linestyle='None', label='$W$/ $upper$ $limits$') # residuals w/ upper limits
	elif (xdataerr3[j] == error_filt) and (j==3):
		ax101.plot(xlogline_C[j], resid_C0[j], color='orange', marker='o', markersize=5, linestyle='None', label='$W$/ $upper$ $limits$') # residuals w/ upper limits
	elif (xdataerr3[j] == error_filt): 
		ax101.plot(xlogline_C[j], resid_C0[j], color='orange', marker='<', markersize=5, linestyle='None') # residuals w/ upper limits

# No upper limits on this plot, just need to sort out legend
ax102.plot(xlogline_maC[4], resid_maC0[4], color='purple', marker='o', markersize=5, linestyle='None', label='$W$/$o$ $upper$ $limits$') # residuals w/o upper limits
ax102.plot(xlogline_maC, resid_maC0, color='purple', marker='o', markersize=5, linestyle='None') # residuals w/o upper limits

ax101.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax102.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline

# Labelling
ax100.set_title('M=0')
ax100.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax101.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.setp(ax101.get_yticklabels()[0], visible=False)
plt.setp(ax101.get_yticklabels()[-1], visible=False)
plt.setp(ax102.get_yticklabels()[0], visible=False)
plt.setp(ax102.get_yticklabels()[-1], visible=False)
ax102.set_xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax100.set_ylabel('$log_{10}(M_{g,CIG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax101.set_ylabel('$Resid$', fontsize=16)
ax102.set_ylabel('$Resid$', fontsize=16)
ax100.legend(loc="best", numpoints=1)
ax101.legend(loc="best", numpoints=1)
ax102.legend(loc="best", numpoints=1)
ax100.set_xlim(10,12)
ax100.set_ylim(-1.0,1.4)
ax101.set_ylim(-0.8,1.0)
ax102.set_ylim(-0.8,1.0)


################################################################################
### Fitting straight lines to logspace data, separated by redshift #############
################################################################################
# Seeing as the lines are very similar for the data with and without upper-limits,
# we may as well keep all the data in for now when splitting by redshift
# We want to fit the lines in a very similar way as before but instead separate
# the data by redshift
# Append data to a list for use later - go in order of lowz, midz, highz
# We also want to fit gradient = 0 lines by redshift
# MAGPHYS
logline_byz_M = []
loglp_byz_M = []
logline_byz_M0 = []
loglp_byz_M0 = []
for i in (0,1,2):
	# Looping over, doing the fit
	logline_z = np.polyfit(dd_sco[colnames[i]+'_M_log'], dd_magphys[colnames[i]+'-sco_log'], 1)
	loglp_z = np.poly1d(logline_z)
	logline_z0 = np.polyfit(dd_sco[colnames[i]+'_M_log'], dd_magphys[colnames[i]+'-sco_log'], 0)
	loglp_z0 = np.poly1d(logline_z0)
	# Saving the data for later
	logline_byz_M.append(logline_z)
	loglp_byz_M.append(loglp_z)
	logline_byz_M0.append(logline_z0)
	loglp_byz_M0.append(loglp_z0)

# CIGALE
logline_byz_C = []
loglp_byz_C = []
logline_byz_C0 = []
loglp_byz_C0 = []
for i in (0,1,2):
	# Looping over, doing the fit
	logline_z = np.polyfit(dd_sco[colnames[i]+'_C_log'], dd_cigale[colnames[i]+'-sco_log'], 1)
	loglp_z = np.poly1d(logline_z)
	logline_z0 = np.polyfit(dd_sco[colnames[i]+'_C_log'], dd_cigale[colnames[i]+'-sco_log'], 0)
	loglp_z0 = np.poly1d(logline_z0)
	# Saving the data for later
	logline_byz_C.append(logline_z)
	loglp_byz_C.append(loglp_z)
	logline_byz_C0.append(logline_z0)
	loglp_byz_C0.append(loglp_z0)


# We also want to calculate the residuals 
resid_byz_M0 = []
resid_byz_C0 = []
for i in (0,1,2):
	# Calculating residual
	resid_byzM0 = dd_magphys[colnames[i]+'-sco_log'] - loglp_byz_M0[i](dd_sco[colnames[i]+'_M_log'])
	resid_byzC0 = dd_cigale[colnames[i]+'-sco_log'] - loglp_byz_C0[i](dd_sco[colnames[i]+'_C_log'])
	# Saving the data
 	resid_byz_M0.append(resid_byzM0)
	resid_byz_C0.append(resid_byzC0)

# We want to calculate the dispersions too for the three lots of residuals
# This is the function
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
disper_byz_M0 = []
disper_byz_C0 = []

for i in (0,1,2):
	disper_byz_M0.append(dispersion(resid_byz_M0[i])) # magphys data
	disper_byz_C0.append(dispersion(resid_byz_C0[i])) # cigale data

# Fit lines to the residuals
residline_byz_M0 = []
residlp_byz_M0   = []
residline_byz_C0 = []
residlp_byz_C0   = []
for m in (0,1,2):
	# Magphys
	lineresid_M0 = np.polyfit(dd_sco[colnames[m]+'_M_log'], resid_byz_M0[m], 1)
	residline_byz_M0.append(lineresid_M0)
	residlp_byz_M0.append(np.poly1d(lineresid_M0))
	# Cigale
	lineresid_C0 = np.polyfit(dd_sco[colnames[m]+'_C_log'], resid_byz_C0[m], 1)
	residline_byz_C0.append(lineresid_C0)
	residlp_byz_C0.append(np.poly1d(lineresid_C0))


# Now we can plot these lines as well 
# MAGPHYS
plt.figure(11)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
plt.plot(np.linspace(10,12,1000), loglp_M(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit all
plt.plot(np.linspace(10,12,1000), loglp_byz_M[0](np.linspace(10,12,1000)), '--', color='b', linewidth=2, label=labelnames[0]) # data fit lowz
plt.plot(np.linspace(10,12,1000), loglp_byz_M[1](np.linspace(10,12,1000)), '--', color='g', linewidth=2, label=labelnames[1]) # data fit midz
plt.plot(np.linspace(10,12,1000), loglp_byz_M[2](np.linspace(10,12,1000)), '--', color='r', linewidth=2, label=labelnames[2]) # data fit highz
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,MAG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)


##
### MAGPHYS - m=0
##
fig12, (ax120, ax121, ax122, ax123) = plt.subplots(4,1, sharex=True, gridspec_kw = {'height_ratios': [4,1,1,1]})
fig12.subplots_adjust(hspace=0)
# Main figure plotting
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = dd_magphys[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_M']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			ax120.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			ax120.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			ax120.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
ax120.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax120.plot(np.zeros(1), np.zeros(1), 'o', color='white', markersize=0, label=' ')
ax120.plot(np.linspace(10,12,1000), loglp_M0(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
ax120.plot(np.zeros(1), np.zeros(1), 'x', color='white', markersize=0, label=' ')
ax120.plot(np.linspace(10,12,1000), loglp_byz_M0[0](np.linspace(10,12,1000)), '--', color='b', linewidth=2, label=labelnames[0]) # data fit lowz
ax120.plot(np.linspace(10,12,1000), loglp_byz_M0[1](np.linspace(10,12,1000)), '--', color='g', linewidth=2, label=labelnames[1]) # data fit midz
ax120.plot(np.linspace(10,12,1000), loglp_byz_M0[2](np.linspace(10,12,1000)), '--', color='r', linewidth=2, label=labelnames[2]) # data fit highz

# Residual figure plotting
# Remember to plot the triangles for upper limits
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_M_log']
	ydata = resid_byz_M0[i]
	xdataerr = dd_sco[colnames[i+3]+'_M']
	# Defining the axes for plotting
	if i == 0:
		axresid = ax121
	elif i == 1:
		axresid = ax122
	elif i == 2:
		axresid = ax123
	# Plotting zero line
	axresid.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
	# Labelling
	plt.setp(axresid.get_yticklabels()[0], visible=False)
	plt.setp(axresid.get_yticklabels()[-1], visible=False)
	axresid.set_ylabel('$Resid$', fontsize=16)
	axresid.legend(loc="upper right", ncol=4, numpoints=1)
	axresid.set_ylim(-1.0,1.0)
	# Annotating the dispersion
	axresid.annotate('$\Delta_{Total}$'+' = '+str(disper_byz_M0[i]), xy = (0.8,0.75), xycoords='axes fraction', fontsize=14)
	# Plotting the fit to the residuals
	axresid.plot(np.linspace(10,12,1000), residlp_byz_M0[i](np.linspace(10,12,1000)), linestyle='-.', color='slategrey', linewidth=2)

# Labelling
ax120.set_title('M=0')
ax120.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax121.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax122.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax123.set_xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax120.set_ylabel('$log_{10}(M_{g,MAG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=18)
ax120.legend(loc="upper right", ncol=3, numpoints=1)
ax120.set_xlim(10,12)
ax120.set_ylim(-1.0,1.4)


# Now we can plot these lines as well 
# CIGALE
plt.figure(13)
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==3):
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			plt.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
plt.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
plt.plot(np.linspace(10,12,1000), loglp_C(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit all
plt.plot(np.linspace(10,12,1000), loglp_byz_C[0](np.linspace(10,12,1000)), '--', color='b', linewidth=2, label=labelnames[0]) # data fit lowz
plt.plot(np.linspace(10,12,1000), loglp_byz_C[1](np.linspace(10,12,1000)), '--', color='g', linewidth=2, label=labelnames[1]) # data fit midz
plt.plot(np.linspace(10,12,1000), loglp_byz_C[2](np.linspace(10,12,1000)), '--', color='r', linewidth=2, label=labelnames[2]) # data fit highz
plt.legend(loc="best", numpoints=1)
plt.xlim(10,12)
plt.ylim(-1.0,1.4)
plt.xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
plt.ylabel('$log_{10}(M_{g,CIG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)


##
### CIGALE - m=0
##
fig14, (ax140, ax141, ax142, ax143) = plt.subplots(4,1, sharex=True, gridspec_kw = {'height_ratios': [4,1,1,1]})
fig14.subplots_adjust(hspace=0)
# Main figure plotting
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = dd_cigale[colnames[i]+'-sco_log']
	xdataerr = dd_sco[colnames[i+3]+'_C']
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			ax140.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			ax140.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==3):
			ax140.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			ax140.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
ax140.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
ax140.plot(np.zeros(1), np.zeros(1), 'o', color='white', markersize=0, label=' ')
ax140.plot(np.linspace(10,12,1000), loglp_C0(np.linspace(10,12,1000)), '-', color='orange', linewidth=2, label='$W$/ $upper$ $limits$') # data fit
ax140.plot(np.zeros(1), np.zeros(1), 'x', color='white', markersize=0, label=' ')
ax140.plot(np.linspace(10,12,1000), loglp_byz_C0[0](np.linspace(10,12,1000)), '--', color='b', linewidth=2, label=labelnames[0]) # data fit lowz
ax140.plot(np.linspace(10,12,1000), loglp_byz_C0[1](np.linspace(10,12,1000)), '--', color='g', linewidth=2, label=labelnames[1]) # data fit midz
ax140.plot(np.linspace(10,12,1000), loglp_byz_C0[2](np.linspace(10,12,1000)), '--', color='r', linewidth=2, label=labelnames[2]) # data fit highz

# Residual figure plotting
# Remember to plot the triangles for upper limits
for i in (0,1,2):
	xdata = dd_sco[colnames[i]+'_C_log']
	ydata = resid_byz_C0[i]
	xdataerr = dd_sco[colnames[i+3]+'_C']
	# Defining the axes for plotting
	if i == 0:
		axresid = ax141
	elif i == 1:
		axresid = ax142
	elif i == 2:
		axresid = ax143
	# Plotting zero line
	axresid.plot(np.linspace(10,12,1000), np.zeros(1000), '--', color='k', linewidth=2) # zeroline
	for j in range(0, len(xdata)):
		# Plot different symbols for upper limits
		if (xdataerr[j] != error_filt) and (j!=3): 
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None')
		elif (xdataerr[j] != error_filt) and (j==3):
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt) and (j==3):
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='o', markersize=5, linestyle='None', label=labelnames[i])
		elif (xdataerr[j] == error_filt): 
			axresid.plot(xdata[j], ydata[j], color=colpoint[i], marker='<', markersize=5, linestyle='None')
	# Labelling
	plt.setp(axresid.get_yticklabels()[0], visible=False)
	plt.setp(axresid.get_yticklabels()[-1], visible=False)
	axresid.set_ylabel('$Resid$', fontsize=16)
	axresid.legend(loc="upper right", ncol=4, numpoints=1)
	axresid.set_ylim(-1.0,1.0)
	# Annotating the graph with the dispersion
	axresid.annotate('$\Delta_{Total}$'+' = '+str(disper_byz_C0[i]), xy = (0.8,0.75), xycoords='axes fraction', fontsize=14)
	# Plotting the fit to the residuals
	axresid.plot(np.linspace(10,12,1000), residlp_byz_C0[i](np.linspace(10,12,1000)), linestyle='-.', color='slategrey', linewidth=2)

# Labelling
ax140.set_title('M=0')
ax140.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax141.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax142.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax143.set_xlabel('$log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=20)
ax140.set_ylabel('$log_{10}(M_{g,CIG})$ - $log_{10}(M_{g,Sco16})$ [$M_{\odot}$]', fontsize=18)
ax140.legend(loc="upper right", ncol=3, numpoints=1)
ax140.set_xlim(10,12)
ax140.set_ylim(-1.0,1.4)



plt.show('all')




