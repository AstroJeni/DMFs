"""
This code reads in the data sent to us by Nick Scoville and puts it into a more
computer friendly format.
"""
################################################################################
### Import various modules #####################################################
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
### Import data ################################################################
################################################################################
### Data path
dat_path = "/home/prongs/c1124553/PhD_Year2/Cosmos_DMF/From_Nick_Sco/"
### File name
fil_name = "Scoville17_sources.txt"
### Read in different sections of the data file
sl = 5 # Starting line
dataRA             = np.genfromtxt(dat_path+fil_name, skip_header = sl, max_rows = 2, dtype=None)
dataparams         = np.genfromtxt(dat_path+fil_name, skip_header = sl+2, max_rows = 2, dtype=None)
dataALMA_RA_1      = np.genfromtxt(dat_path+fil_name, skip_header = sl+5, max_rows = 1, dtype=None)
dataALMA_flx_1     = np.genfromtxt(dat_path+fil_name, skip_header = sl+6, max_rows = 2, dtype=None)
dataALMA_flx_use_1 = np.genfromtxt(dat_path+fil_name, skip_header = sl+8, max_rows = 1, dtype=None)
dataALMA_obs_1     = np.genfromtxt(dat_path+fil_name, skip_header = sl+11, max_rows = 1, dtype=None)
### Sometimes we have two ALMA observations, and we need to account for this
test             = np.genfromtxt(dat_path+fil_name, skip_header = sl+12, max_rows = 1, dtype=None)


### Extract the data we need line by line
colnames = []
datasave = []
### Source ID, RA, Dec, z
colnames.append(list(dataRA[0]))
datasave.append(list(dataRA[1].astype(float)))
### Various parameters calculated by SED fitting
colnames.append(list(dataparams[0]))
datasave.append(list(dataparams[1].astype(float)))
### ALMA RA Dec and offsets
colnames.append(['ALMA_freq(GHz)_1', 'ALMA_RA_1', 'ALMA_Dec_1', 'ALMA_offsetRA(")_1',
                 'ALMA_offsetDec(")_1'])
### Some inconsistent formatting to deal with
try:
	datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
	                 float(dataALMA_RA_1[()][7][0:-1]), float(dataALMA_RA_1[()][8][0:-1])])
except:
	datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
	                 float(dataALMA_RA_1[()][6][1:-1]), float(dataALMA_RA_1[()][7][0:-1])])
### ALMA peak and total fluxes
colnames.append(['ALMA_peakflux(mJy)_1', 'ALMA_peakSNR_1', 'ALMA_peakRA_1', 'ALMA_peakDec_1'])
datasave.append([dataALMA_flx_1[0][2], dataALMA_flx_1[0][5], dataALMA_flx_1[0][8], 
                 dataALMA_flx_1[0][11]])
colnames.append(['ALMA_totflux(mJy)_1', 'ALMA_totSNR_1', 'ALMA_totRA_1', 'ALMA_totDec_1'])
datasave.append([dataALMA_flx_1[1][2], dataALMA_flx_1[1][5], dataALMA_flx_1[1][8],
                 dataALMA_flx_1[1][11]])
### ALMA flux used
colnames.append(['ALMA_useflux(mJy)_1', 'ALMA_useSNR_1', 'ALMA_use_pbcor_1', 'ALMA_use_ftom_1'])
### Some inconsistent formatting to deal with
try:
	datasave.append([dataALMA_flx_use_1[()][3], dataALMA_flx_use_1[()][6], dataALMA_flx_use_1[()][9], 
	                 dataALMA_flx_use_1[()][12]])
except:
	datasave.append([dataALMA_flx_use_1[()][3], float(dataALMA_flx_use_1[()][5][1:]), 
                     dataALMA_flx_use_1[()][8], dataALMA_flx_use_1[()][11]])
### ALMA observation
colnames.append(['ALMAbeam_a(")_1','ALMAbeam_b(")_1','rms(mJy)_1','obs_date_1'])
datasave.append([dataALMA_obs_1[()][2], dataALMA_obs_1[()][4], dataALMA_obs_1[()][8], 
                 dataALMA_obs_1[()][13]])


### Testing to see if we have one or two ALMA observations
try:
	test[0] == 'Source'
	### Then we do not have a second ALMA observation and we are moving onto the next source. So, we
    ### need to put the data for the second ALMA source as blank
	### ALMA RA Dec and offsets
	colnames.append(['ALMA_freq(GHz)_2', 'ALMA_RA_2', 'ALMA_Dec_2', 'ALMA_offsetRA(")_2',
		             'ALMA_offsetDec(")_2'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA peak and total fluxes
	colnames.append(['ALMA_peakflux(mJy)_2', 'ALMA_peakSNR_2', 'ALMA_peakRA_2', 'ALMA_peakDec_2'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	colnames.append(['ALMA_totflux(mJy)_2', 'ALMA_totSNR_2', 'ALMA_totRA_2', 'ALMA_totDec_2'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA flux used
	colnames.append(['ALMA_useflux(mJy)_2', 'ALMA_useSNR_2', 'ALMA_use_pbcor_2', 'ALMA_use_ftom_2'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA observation
	colnames.append(['ALMAbeam_a(")_2','ALMAbeam_b(")_2','rms(mJy)_2','obs_date_2'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])

	# If we do not have a second ALMA source then we do not have a third one either
	### ALMA RA Dec and offsets
	colnames.append(['ALMA_freq(GHz)_3', 'ALMA_RA_3', 'ALMA_Dec_3', 'ALMA_offsetRA(")_3',
		             'ALMA_offsetDec(")_3'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA peak and total fluxes
	colnames.append(['ALMA_peakflux(mJy)_3', 'ALMA_peakSNR_3', 'ALMA_peakRA_3', 'ALMA_peakDec_3'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	colnames.append(['ALMA_totflux(mJy)_3', 'ALMA_totSNR_3', 'ALMA_totRA_3', 'ALMA_totDec_3'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA flux used
	colnames.append(['ALMA_useflux(mJy)_3', 'ALMA_useSNR_3', 'ALMA_use_pbcor_3', 'ALMA_use_ftom_3'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
	### ALMA observation
	colnames.append(['ALMAbeam_a(")_3','ALMAbeam_b(")_3','rms(mJy)_3','obs_date_3'])
	datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])

	# Now we need a new starting line for the next source
	sl = sl + 14

except IndexError:
	### Then this means we have a second ALMA source and we need to extract the data and save it
	### Extract the data
	dataALMA_RA_2      = np.genfromtxt(dat_path+fil_name, skip_header = sl+12, max_rows = 1, dtype=None)
	dataALMA_flx_2     = np.genfromtxt(dat_path+fil_name, skip_header = sl+13, max_rows = 2, dtype=None)
	dataALMA_flx_use_2 = np.genfromtxt(dat_path+fil_name, skip_header = sl+15, max_rows = 1, dtype=None)
	dataALMA_obs_2     = np.genfromtxt(dat_path+fil_name, skip_header = sl+18, max_rows = 1, dtype=None)
	### ALMA RA Dec and offsets
	colnames.append(['ALMA_freq(GHz)_2', 'ALMA_RA_2', 'ALMA_Dec_2', 'ALMA_offsetRA(")_2',
		             'ALMA_offsetDec(")_2'])
	### Some inconsistent formatting to deal with
	try:
		datasave.append([dataALMA_RA_2[()][0],  dataALMA_RA_2[()][3], dataALMA_RA_2[()][4], 
			             float(dataALMA_RA_2[()][7][0:-1]), float(dataALMA_RA_2[()][8][0:-1])])
	except:
		datasave.append([dataALMA_RA_2[()][0],  dataALMA_RA_2[()][3], dataALMA_RA_2[()][4], 
			             float(dataALMA_RA_2[()][6][1:-1]), float(dataALMA_RA_2[()][7][0:-1])])
	### ALMA peak and total fluxes
	colnames.append(['ALMA_peakflux(mJy)_2', 'ALMA_peakSNR_2', 'ALMA_peakRA_2', 'ALMA_peakDec_2'])
	datasave.append([dataALMA_flx_2[0][2], dataALMA_flx_2[0][5], dataALMA_flx_2[0][8], 
		             dataALMA_flx_2[0][11]])
	colnames.append(['ALMA_totflux(mJy)_2', 'ALMA_totSNR_2', 'ALMA_totRA_2', 'ALMA_totDec_2'])
	datasave.append([dataALMA_flx_2[1][2], dataALMA_flx_2[1][5], dataALMA_flx_2[1][8], 
                     dataALMA_flx_2[1][11]])
	### ALMA flux used
	colnames.append(['ALMA_useflux(mJy)_2', 'ALMA_useSNR_2', 'ALMA_use_pbcor_2', 'ALMA_use_ftom_2'])

	### Some inconsistent formatting to deal with
	try:
		datasave.append([dataALMA_flx_use_2[()][3], dataALMA_flx_use_2[()][6], dataALMA_flx_use_2[()][9], 
			             dataALMA_flx_use_2[()][12]])
	except:
		datasave.append([dataALMA_flx_use_2[()][3], float(dataALMA_flx_use_2[()][5][1:]), 
		                 dataALMA_flx_use_2[()][8], dataALMA_flx_use_2[()][11]])
	### ALMA observation
	colnames.append(['ALMAbeam_a(")_2','ALMAbeam_b(")_2','rms(mJy)_2','obs_date_2'])
	datasave.append([dataALMA_obs_2[()][2], dataALMA_obs_2[()][4], dataALMA_obs_2[()][8], 
		             dataALMA_obs_2[()][13]])

	# Now we need a new starting line for the next source
	sl = sl + 20

	# We need to check for a third ALMA observation
	test2 = np.genfromtxt(dat_path+fil_name, skip_header = sl-1, max_rows = 1, dtype=None)

	try:
		test2[0] == 'Source'
		### Then we do not have a third ALMA observation and we are moving onto the next source. So, we
		### need to put the data for the second ALMA source as blank
		### ALMA RA Dec and offsets
		colnames.append(['ALMA_freq(GHz)_3', 'ALMA_RA_3', 'ALMA_Dec_3', 'ALMA_offsetRA(")_3',
				         'ALMA_offsetDec(")_3'])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA peak and total fluxes
		colnames.append(['ALMA_peakflux(mJy)_3', 'ALMA_peakSNR_3', 'ALMA_peakRA_3', 'ALMA_peakDec_3'])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		colnames.append(['ALMA_totflux(mJy)_3', 'ALMA_totSNR_3', 'ALMA_totRA_3', 'ALMA_totDec_3'])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA flux used
		colnames.append(['ALMA_useflux(mJy)_3', 'ALMA_useSNR_3', 'ALMA_use_pbcor_3', 'ALMA_use_ftom_3'])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA observation
		colnames.append(['ALMAbeam_a(")_3','ALMAbeam_b(")_3','rms(mJy)_3','obs_date_3'])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])

		sl = sl + 1

	except IndexError:
		### Then this means we have a second ALMA source and we need to extract the data and save it
		### Extract the data
		dataALMA_RA_3      = np.genfromtxt(dat_path+fil_name, skip_header = sl+12, max_rows = 1, dtype=None)
		dataALMA_flx_3     = np.genfromtxt(dat_path+fil_name, skip_header = sl+13, max_rows = 2, dtype=None)
		dataALMA_flx_use_3 = np.genfromtxt(dat_path+fil_name, skip_header = sl+15, max_rows = 1, dtype=None)
		dataALMA_obs_3     = np.genfromtxt(dat_path+fil_name, skip_header = sl+18, max_rows = 1, dtype=None)
		### ALMA RA Dec and offsets
		colnames.append(['ALMA_freq(GHz)_3', 'ALMA_RA_3', 'ALMA_Dec_3', 'ALMA_offsetRA(")_3',
				         'ALMA_offsetDec(")_3'])
		### Some inconsistent formatting to deal with
		try:
			datasave.append([dataALMA_RA_3[()][0],  dataALMA_RA_3[()][3], dataALMA_RA_3[()][4], 
					         float(dataALMA_RA_3[()][7][0:-1]), float(dataALMA_RA_3[()][8][0:-1])])
		except:
			datasave.append([dataALMA_RA_3[()][0],  dataALMA_RA_3[()][3], dataALMA_RA_3[()][4], 
					         float(dataALMA_RA_3[()][6][1:-1]), float(dataALMA_RA_3[()][7][0:-1])])
		### ALMA peak and total fluxes
		colnames.append(['ALMA_peakflux(mJy)_3', 'ALMA_peakSNR_3', 'ALMA_peakRA_3', 'ALMA_peakDec_3'])
		datasave.append([dataALMA_flx_3[0][2], dataALMA_flx_3[0][5], dataALMA_flx_3[0][8], 
				         dataALMA_flx_3[0][11]])
		colnames.append(['ALMA_totflux(mJy)_3', 'ALMA_totSNR_3', 'ALMA_totRA_3', 'ALMA_totDec_3'])
		datasave.append([dataALMA_flx_3[1][2], dataALMA_flx_3[1][5], dataALMA_flx_3[1][8], 
		                 dataALMA_flx_3[1][11]])
		### ALMA flux used
		colnames.append(['ALMA_useflux(mJy)_3', 'ALMA_useSNR_3', 'ALMA_use_pbcor_3', 'ALMA_use_ftom_3'])

		### Some inconsistent formatting to deal with
		try:
			datasave.append([dataALMA_flx_use_3[()][3], dataALMA_flx_use_3[()][6], dataALMA_flx_use_3[()][9], 
					         dataALMA_flx_use_3[()][12]])
		except:
			datasave.append([dataALMA_flx_use_3[()][3], float(dataALMA_flx_use_3[()][5][1:]), 
				             dataALMA_flx_use_3[()][8], dataALMA_flx_use_3[()][11]])
		### ALMA observation
		colnames.append(['ALMAbeam_a(")_3','ALMAbeam_b(")_3','rms(mJy)_3','obs_date_3'])
		datasave.append([dataALMA_obs_3[()][2], dataALMA_obs_3[()][4], dataALMA_obs_3[()][8], 
				         dataALMA_obs_3[()][13]])

		sl = sl + 8

### Concatenate the lists so that for the final data file, one row = one list
colnames_f = [item for sublist in colnames for item in sublist]
datasave_fin = [item for sublist in datasave for item in sublist]
datasave_f = []
datasave_f.append(datasave_fin)







# Now we have done this, we can loop over the total number of sources (797)
for i in range(1, 708): #798
	### Read in different sections of the data file
	dataRA             = np.genfromtxt(dat_path+fil_name, skip_header = sl, max_rows = 2, dtype=None)
	dataparams         = np.genfromtxt(dat_path+fil_name, skip_header = sl+2, max_rows = 2, dtype=None)
	dataALMA_RA_1      = np.genfromtxt(dat_path+fil_name, skip_header = sl+5, max_rows = 1, dtype=None)
	dataALMA_flx_1     = np.genfromtxt(dat_path+fil_name, skip_header = sl+6, max_rows = 2, dtype=None)
	dataALMA_flx_use_1 = np.genfromtxt(dat_path+fil_name, skip_header = sl+8, max_rows = 1, dtype=None)
	dataALMA_obs_1     = np.genfromtxt(dat_path+fil_name, skip_header = sl+11, max_rows = 1, dtype=None)
	### Sometimes we have two ALMA observations, and we need to account for this
	try:	
		test               = np.genfromtxt(dat_path+fil_name, skip_header = sl+12, max_rows = 1, dtype=None)
	except:
		# We have reached the end of the file 
		### Extract the data we need line by line
		datasave = []
		### Source ID, RA, Dec, z
		datasave.append(list(dataRA[1].astype(float)))
		### Various parameters calculated by SED fitting
		datasave.append(list(dataparams[1].astype(float)))
		### ALMA RA Dec and offsets
		### Some inconsistent formatting to deal with
		if i != 656:
			try:
				datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
							     float(dataALMA_RA_1[()][7][0:-1]), float(dataALMA_RA_1[()][8][0:-1])])
			except:
				datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
							     float(dataALMA_RA_1[()][6][1:-1]), float(dataALMA_RA_1[()][7][0:-1])])
		else:
			try:
				datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
							     float(dataALMA_RA_1[()][7][0:-1]), dataALMA_RA_1[()][8][0:-1]])
			except:
				datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
							     float(dataALMA_RA_1[()][6][1:-1]), dataALMA_RA_1[()][7][0:-1]])
		### ALMA peak and total fluxes
		datasave.append([dataALMA_flx_1[0][2], dataALMA_flx_1[0][5], dataALMA_flx_1[0][8], 
				         dataALMA_flx_1[0][11]])
		datasave.append([dataALMA_flx_1[1][2], dataALMA_flx_1[1][5], dataALMA_flx_1[1][8],
				         dataALMA_flx_1[1][11]])
		### ALMA flux used
		### Some inconsistent formatting to deal with
		try:
			datasave.append([dataALMA_flx_use_1[()][3], dataALMA_flx_use_1[()][6], dataALMA_flx_use_1[()][9], 
					         dataALMA_flx_use_1[()][12]])
		except:
			datasave.append([dataALMA_flx_use_1[()][3], float(dataALMA_flx_use_1[()][5][1:]), 
				             dataALMA_flx_use_1[()][8], dataALMA_flx_use_1[()][11]])
		### ALMA observation
		datasave.append([dataALMA_obs_1[()][2], dataALMA_obs_1[()][4], dataALMA_obs_1[()][8], 
				         dataALMA_obs_1[()][13]])

		# Concatenate the lists so that for the final data file, one row = one list
		datasave_fin = [item for sublist in datasave for item in sublist]
		datasave_f.append(datasave_fin)

		
	### Extract the data we need line by line
	datasave = []
	### Source ID, RA, Dec, z
	datasave.append(list(dataRA[1].astype(float)))
	### Various parameters calculated by SED fitting
	datasave.append(list(dataparams[1].astype(float)))
	### ALMA RA Dec and offsets
	### Some inconsistent formatting to deal with
	if i != 656:
		try:
			datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
					         float(dataALMA_RA_1[()][7][0:-1]), float(dataALMA_RA_1[()][8][0:-1])])
		except:
			datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
					         float(dataALMA_RA_1[()][6][1:-1]), float(dataALMA_RA_1[()][7][0:-1])])
	else:
		try:
			datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
					         float(dataALMA_RA_1[()][7][0:-1]), dataALMA_RA_1[()][8][0:-1]])
		except:
			datasave.append([dataALMA_RA_1[()][0],  dataALMA_RA_1[()][3], dataALMA_RA_1[()][4], 
					         float(dataALMA_RA_1[()][6][1:-1]), dataALMA_RA_1[()][7][0:-1]])
	### ALMA peak and total fluxes
	datasave.append([dataALMA_flx_1[0][2], dataALMA_flx_1[0][5], dataALMA_flx_1[0][8], 
		             dataALMA_flx_1[0][11]])
	datasave.append([dataALMA_flx_1[1][2], dataALMA_flx_1[1][5], dataALMA_flx_1[1][8],
		             dataALMA_flx_1[1][11]])
	### ALMA flux used
	### Some inconsistent formatting to deal with
	try:
		datasave.append([dataALMA_flx_use_1[()][3], dataALMA_flx_use_1[()][6], dataALMA_flx_use_1[()][9], 
			             dataALMA_flx_use_1[()][12]])
	except:
		datasave.append([dataALMA_flx_use_1[()][3], float(dataALMA_flx_use_1[()][5][1:]), 
		                 dataALMA_flx_use_1[()][8], dataALMA_flx_use_1[()][11]])
	### ALMA observation
	datasave.append([dataALMA_obs_1[()][2], dataALMA_obs_1[()][4], dataALMA_obs_1[()][8], 
		             dataALMA_obs_1[()][13]])


	### Testing to see if we have one or two ALMA observations
	try:
		test[0] == 'Source'
		### Then we do not have a second ALMA observation and we are moving onto the next source. So, we
		### need to put the data for the second ALMA source as blank
		### ALMA RA Dec and offsets
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA peak and total fluxes
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA flux used
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
		### ALMA observation
		datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])

		print(' There is no second ALMA source for: '), 
		print(datasave[0][0])
		# Now we need a new starting line for the next source
		sl = sl + 14

	except IndexError:
		### Then this means we have a second ALMA source and we need to extract the data and save it
		### Extract the data
		dataALMA_RA_2      = np.genfromtxt(dat_path+fil_name, skip_header = sl+5+7, max_rows = 1, dtype=None)
		dataALMA_flx_2     = np.genfromtxt(dat_path+fil_name, skip_header = sl+6+7, max_rows = 2, dtype=None)
		dataALMA_flx_use_2 = np.genfromtxt(dat_path+fil_name, skip_header = sl+8+7, max_rows = 1, dtype=None)
		dataALMA_obs_2     = np.genfromtxt(dat_path+fil_name, skip_header = sl+11+7, max_rows = 1, dtype=None)
		
		### ALMA RA Dec and offsets
		### Some inconsistent formatting to deal with
		try:
			datasave.append([dataALMA_RA_2[()][0],  dataALMA_RA_2[()][3], dataALMA_RA_2[()][4], 
					         float(dataALMA_RA_2[()][7][0:-1]), float(dataALMA_RA_2[()][8][0:-1])])
		except:
			datasave.append([dataALMA_RA_2[()][0],  dataALMA_RA_2[()][3], dataALMA_RA_2[()][4], 
					         float(dataALMA_RA_2[()][6][1:-1]), float(dataALMA_RA_2[()][7][0:-1])])
		### ALMA peak and total fluxes
		datasave.append([dataALMA_flx_2[0][2], dataALMA_flx_2[0][5], dataALMA_flx_2[0][8], 
				         dataALMA_flx_2[0][11]])
		datasave.append([dataALMA_flx_2[1][2], dataALMA_flx_2[1][5], dataALMA_flx_2[1][8], 
		                 dataALMA_flx_2[1][11]])
		### ALMA flux used
		### Some inconsistent formatting to deal with
		try:
			datasave.append([dataALMA_flx_use_2[()][3], dataALMA_flx_use_2[()][6], dataALMA_flx_use_2[()][9], 
					         dataALMA_flx_use_2[()][12]])
		except:
			datasave.append([dataALMA_flx_use_2[()][3], float(dataALMA_flx_use_2[()][5][1:]), 
				             dataALMA_flx_use_2[()][8], dataALMA_flx_use_2[()][11]])
		### ALMA observation
		datasave.append([dataALMA_obs_2[()][2], dataALMA_obs_2[()][4], dataALMA_obs_2[()][8], 
				         dataALMA_obs_2[()][13]])

		# Now we need a new starting line for the next source
		sl = sl + 20

		print(' There is a second ALMA source for: '), 
		print(datasave[0][0])

		# We need to check for a third ALMA observation
		test2 = np.genfromtxt(dat_path+fil_name, skip_header = sl-1, max_rows = 1, dtype=None)

		try:
			test2[0] == 'Source'
			### Then we do not have a third ALMA observation and we are moving onto the next source. So, we
			### need to put the data for the second ALMA source as blank
			### ALMA RA Dec and offsets
			datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')])
			### ALMA peak and total fluxes
			datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
			datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
			### ALMA flux used
			datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
			### ALMA observation
			datasave.append([float('NaN'), float('NaN'), float('NaN'), float('NaN')])
			
			sl = sl + 1

			print(' There is a no third ALMA source for: '), 
			print(datasave[0][0])

		except IndexError:
			### Then this means we have a third ALMA source and we need to extract the data and save it
			### Extract the data
			dataALMA_RA_3      = np.genfromtxt(dat_path+fil_name, skip_header = sl-1, max_rows = 1, dtype=None)
			dataALMA_flx_3     = np.genfromtxt(dat_path+fil_name, skip_header = sl, max_rows = 2, dtype=None)
			dataALMA_flx_use_3 = np.genfromtxt(dat_path+fil_name, skip_header = sl+2, max_rows = 1, dtype=None)
			dataALMA_obs_3    = np.genfromtxt(dat_path+fil_name, skip_header = sl+5, max_rows = 1, dtype=None)
		
			### ALMA RA Dec and offsets
			### Some inconsistent formatting to deal with
			try:
				datasave.append([dataALMA_RA_3[()][0],  dataALMA_RA_3[()][3], dataALMA_RA_3[()][4], 
							     float(dataALMA_RA_3[()][7][0:-1]), float(dataALMA_RA_3[()][8][0:-1])])
			except:
				datasave.append([dataALMA_RA_3[()][0],  dataALMA_RA_3[()][3], dataALMA_RA_3[()][4], 
							     float(dataALMA_RA_3[()][6][1:-1]), float(dataALMA_RA_3[()][7][0:-1])])
			### ALMA peak and total fluxes
			datasave.append([dataALMA_flx_3[0][2], dataALMA_flx_3[0][5], dataALMA_flx_3[0][8], 
						     dataALMA_flx_3[0][11]])
			datasave.append([dataALMA_flx_3[1][2], dataALMA_flx_3[1][5], dataALMA_flx_3[1][8], 
				             dataALMA_flx_3[1][11]])
			### ALMA flux used
			### Some inconsistent formatting to deal with
			try:
				datasave.append([dataALMA_flx_use_3[()][3], dataALMA_flx_use_3[()][6], dataALMA_flx_use_3[()][9], 
							     dataALMA_flx_use_3[()][12]])
			except:
				datasave.append([dataALMA_flx_use_3[()][3], float(dataALMA_flx_use_3[()][5][1:]), 
						         dataALMA_flx_use_3[()][8], dataALMA_flx_use_3[()][11]])
			### ALMA observation
			datasave.append([dataALMA_obs_3[()][2], dataALMA_obs_3[()][4], dataALMA_obs_3[()][8], 
						     dataALMA_obs_3[()][13]])

			# Now we need a new starting line for the next source
			sl = sl + 8

			print(' There is a third ALMA source for: '), 
			print(datasave[0][0])

	# Concatenate the lists so that for the final data file, one row = one list
	datasave_fin = [item for sublist in datasave for item in sublist]
	datasave_f.append(datasave_fin)





# Save this to a csv file now
# Create a pandas dataframe
df = pd.DataFrame(datasave_f)
df.columns = colnames_f
df.to_csv(dat_path+'Scoville17_sources797_reformatted.csv')










