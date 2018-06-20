"""
Testing LMFIT from python for fitting
"""
import numpy as np
from lmfit import Parameters, Minimizer, conf_interval, conf_interval2d, minimize, printfuncs
import matplotlib.pyplot as plt
from scipy import odr
import sys
plt.close('all')


################################################################################
### Test 1: Standard LMFIT #####################################################
################################################################################
# Set up some fake x and y data
np.random.seed(7)
x_data = np.linspace(0,10,20)
delta = np.random.uniform(-1,1, size=(20,))
y_data = x_data + 2 + delta

# Trend is roughly y = x + 2 i.e. m=1 and c =2

# Plot this fake data
plt.figure()
plt.plot(x_data, y_data, 'ko', ls='None')

# LMFIT needs the residual of the data points from the model
def residual(pars, xdata, ydata=None):
	m = pars['m'].value
	c = pars['c'].value
	model = (m*xdata) + c
	if ydata is None:
		return model
	return model - ydata

params = Parameters() 
params.add('m', value = 1.0)
params.add('c', value = 2.0)
mi = minimize(residual, params, args=(x_data, y_data))

mfitval = mi.params['m'].value
mfiterr = mi.params['m'].stderr

cfitval = mi.params['c'].value
cfiterr = mi.params['c'].stderr

# Checking the fit on the plot
x_plot = np.linspace(min(x_data), max(x_data), 100)
plt.figure()
plt.plot(x_data, y_data, 'ko', ls='None')
plt.plot(x_plot, (mfitval*x_plot + cfitval), 'b')


################################################################################
### Test 2: Include some y-data errors #########################################
################################################################################
# Set up some fake x and y data
np.random.seed(7)
x_data = np.linspace(0,10,20)
delta = np.random.uniform(-1,1, size=(20,))
y_data = x_data + 2 + delta
y_data_err = (np.random.uniform(-1,1, size=(20,)))*2

# Trend is roughly y = x + 2 i.e. m=1 and c =2

# LMFIT needs the residual of the data points from the model
def residual(pars, xdata, ydata, ydataerr):
	m = pars['m'].value
	c = pars['c'].value
	model = (m*xdata) + c

	if ydata is None:
		return model
	else:
		resid = model - ydata
		# Weight the residuals to account for errors
		weight = np.sqrt(resid**2 / ydataerr**2)

	return weight

# Parameters to fit
params = Parameters() 
params.add('m', value = 1.0)
params.add('c', value = 2.0)
mi2 = minimize(residual, params, args=(x_data, y_data, y_data_err))

mfitval2 = mi2.params['m'].value
mfiterr2 = mi2.params['m'].stderr

cfitval2 = mi2.params['c'].value
cfiterr2 = mi2.params['c'].stderr

# Checking the fit on the plot
plt.figure()
plt.errorbar(x_data, y_data, yerr=y_data_err, ls='None', marker='o', color='k')
plt.plot(x_plot, (mfitval*x_plot + cfitval), 'b')
plt.plot(x_plot, (mfitval2*x_plot + cfitval2), 'c')


################################################################################
### Test 3: x-data errors and y-data errors ####################################
################################################################################
# Set up some fake x and y data
np.random.seed(7)
x_data = np.linspace(0,10,20)
delta = np.random.uniform(-1,1, size=(20,))
y_data = x_data + 2 + delta
y_data_err = (np.random.uniform(-1,1, size=(20,)))*2
x_data_err = (np.random.uniform(-1,1, size=(20,)))

# Trend is roughly y = x + 2 i.e. m=1 and c =2

# LMFIT needs the residual of the data points from the model
def residual(pars, xdata, ydata, xdataerr, ydataerr):
	m = pars['m'].value
	c = pars['c'].value
	model = (m*xdata) + c

	if ydata is None:
		return model
	else:
		# sqrt of chi-squared in y 
		resid_y = model - ydata
		# Weight the residuals to account for errors
		weight_y = np.sqrt(resid_y**2 / ydataerr**2)

		# sqrt of chi-squared in x
		resid_x = model - xdata
		# Weight the residuals to account for errors
		weight_x = np.sqrt(resid_x**2 / xdataerr**2)




	return weight

# Parameters to fit
params = Parameters() 
params.add('m', value = 1.0)
params.add('c', value = 2.0)
mi3 = minimize(residual, params, args=(x_data, y_data, x_data_err, y_data_err))

mfitval3 = mi3.params['m'].value
mfiterr3 = mi3.params['m'].stderr

cfitval3 = mi3.params['c'].value
cfiterr3 = mi3.params['c'].stderr

# Checking the fit on the plot
plt.figure()
plt.errorbar(x_data, y_data, xerr=x_data_err, yerr=y_data_err, ls='None', marker='o', color='k')
plt.plot(x_plot, (mfitval*x_plot + cfitval), 'b', label='Standard LMFIT')
plt.plot(x_plot, (mfitval2*x_plot + cfitval2), 'c', label='yerrs LMFIT')
plt.plot(x_plot, (mfitval3*x_plot + cfitval3), 'g', label='x+yerrs LMFIT')
plt.legend(loc='best')






sys.exit()


################################################################################
### Test 4: Include some x-data errors - ODR ###################################
################################################################################
# Set up some fake x and y data
np.random.seed(7)
x_data = np.linspace(0,10,20)
delta = np.random.uniform(-1,1, size=(20,))
y_data = x_data + 2 + delta
y_data_err = (np.random.uniform(-1,1, size=(20,)))*2
x_data_err = (np.random.uniform(-1,1, size=(20,)))

# Trend is roughly y = x + 2 i.e. m=1 and c =2

# In this instance, we are going to use scipy.odr, which is for orthogonal distance
# regression in python
# Really, in order to use least squares fitting in Python, you have to make the 
# assumption that your errors are Gaussian based. This also includes the assumption
# that they are symmetrical about the data point. This is true for the Scoville 
# data, it is not true for the magphys data. So, we have to be a bit cheaty. Really
# we should probably do some kind of Bayesian stats but that's just difficult. So, 
# we do cheating and stick a caveat on the data. The cheaty solution is to instead
# average the up-down errors of the data points, and feed this into the fitting
# routine. We then bootstrap between the real errors and this will give us the errors
# on our slope and intercept. 

# Here, in testing, we will have uniform errors to begin with. Then we will make a 
# few have non-uniform errors afterwards and see how things differ.

# Define the function to fit against
def func(params, x):
	return params[0]*x + params[1]

# Create the model
linear = odr.Model(func)

# Create the data 
mydata = odr.RealData(x_data, y_data, sx = x_data_err, sy = y_data_err)

# Setting up ODR
myodr = odr.ODR(mydata, linear, beta0=[1.0, 2.0])

# Run the fit
output = myodr.run()

# See the output
output.pprint()

# Get out the params
m_fit = output.beta[0]
c_fit = output.beta[1]

m_fiterr = output.sd_beta[0]
c_fiterr = output.sd_beta[1]

# Plot this with the errors
plt.figure()
plt.errorbar(x_data, y_data, xerr=x_data_err, yerr=y_data_err, ls='None', marker='o', color='k')
plt.plot(x_plot, (mfitval*x_plot + cfitval), 'b', label='LMFIT')
plt.plot(x_plot, (mfitval2*x_plot + cfitval2), 'c', label='yerrs LMFIT')
plt.plot(x_plot, (mfitval3*x_plot + cfitval3), 'g', label='x+yerrs LMFIT')
plt.plot(x_plot, (m_fit*x_plot + c_fit), 'm', label='ODR')
#plt.plot(x_plot, ((m_fit+m_fiterr)*x_plot + (c_fit+c_fiterr)), 'm--')
#plt.plot(x_plot, ((m_fit-m_fiterr)*x_plot + (c_fit-c_fiterr)), 'm--')
plt.legend(loc='best')
plt.show('all')






################################################################################
### Test 5: Bootstrapping with non-uniform y errors and uniform x errors########
################################################################################
# Have to fabricate some non-uniform y errors for this bit, so it replicates more
# closely the sort of data I have 
# Set up some fake x and y data
np.random.seed(7)
x_data = np.linspace(0,10,20)
delta = np.random.uniform(-1,1, size=(20,)) # perturbation about straight line
y_data = x_data + 2 + delta
# The x-data can be uniform
x_data_err = (np.random.uniform(-1,1, size=(20,)))

# Setting up the y-data
# Here we have some random y errors, which will be the 16th and 84th percentiles
#  for the purposes of the fake data
np.random.seed(11)
yerr84 = np.random.uniform(0,2, size=(20,))
np.random.seed(16)
yerr16 = np.random.uniform(0,2, size=(20,))

# Then we need to set up the 2.5th and 97.5th percentiles
np.random.seed(25)
yerr97 = yerr84 + np.random.uniform(0,0.5, size=(20,))
np.random.seed(32)
yerr2 = yerr16 + np.random.uniform(0,0.5, size=(20,))

# Plot these to make sure that they are sensible
plt.figure()
plt.errorbar(x_data, y_data, yerr=[yerr16, yerr84], color='blue', marker='o', ls='None')
plt.errorbar(x_data, y_data, yerr=[yerr2, yerr97], color='cyan', marker='o', ls='None')

# This isn't the form that we need this data in as this is not the form of the 
# MAGPHYS data, so we need to get it into this form
magphys50 = y_data
magphys84 = y_data + yerr84
magphys97 = y_data + yerr84 + yerr97
magphys16 = y_data - yerr16 
magphys2  = y_data - yerr16 - yerr2

# Plot this just to check
plt.figure()
plt.plot(x_data, magphys2, 'bo', ls='-')
plt.plot(x_data, magphys16, 'co', ls='-')
plt.plot(x_data, magphys50, 'go', ls='-')
plt.plot(x_data, magphys84, 'mo', ls='-')
plt.plot(x_data, magphys97, 'ro', ls='-')
plt.show('all')

# Now we need to do bootstrapping. We do bootstrapping by randomly sampling the 
# data points within the parameter space and refitting the line. The errors that
# we use are the same as the original data. Usually you would use a Gaussian with
# a s.d. that is your error but this is not possible in this case because the 
# y errors are not uniform. So we have to be a bit contrived and think of something
# reasonable. What we do is pick a random number between 0 and 1 to work out which
# bin we are in i.e. 16th - 50th percentile or 2.5th - 16th percentile. Then we have
# to scale this number to the value of the data point i.e. where along the 16th-50th
# percentile range are we for our values? This is then the new data point. We use
# the same errors as before. We are not dealing with x errors yet. For now, the
# x data point will stay the same.

for i in range(0,100): # This is how many times we want to run the bootstrap
	bootstrap_magphys50 = [] # The new data set from the bootstrap iteration
	bootstrap_xdata = []

	for j in range(0, len(magphys50)): # For each data point, we need a new 50th percentile
		# Pick a random number
		rando = np.random.uniform(2.5,97.5, size=1)
		# Scale this to the range of the given data point
		# To make sure this is representative of the true shape of the pdf (which
		# we don't know, we only have a few percentile values to go off) we need 
		# to make sure that we properly scale i.e. where within this range we are 
		# sampling. So what we do is 
		# [[[chosenpercentile(rando)] - [lowerbinlimit] / [upperbinlimit - lowerbinlimit]] * binwidth ] + lowerbinlimit
		if (rando <= 16): # Bottom bin of pdf
			newdatapoint = (((rando - 2.5)/(16.0 - 2.5))*(magphys16[j]-magphys2[j])) + magphys2[j]

		elif (rando > 16) and (rando <= 50): # Second bin of pdf
			newdatapoint = (((rando - 16.0)/(50.0 - 16.0))*(magphys50[j]-magphys16[j])) + magphys16[j]

		elif (rando > 50) and (rando <= 84): # Third bin of pdf
			newdatapoint = (((rando - 50.0)/(84.0 - 50.0))*(magphys84[j]-magphys50[j])) + magphys50[j]

		elif (rando > 84) and (rando <= 97.5): # Fourth bin of pdf
			newdatapoint = (((rando - 84.0)/(97.5 - 84.0))*(magphys97[j]-magphys84[j])) + magphys84[j]
		# Now we have a new datapoint, we need to save it
		bootstrap_magphys50.append(newdatapoint)
	
	# Now we have a new data array for the 50th percentile fluxes, which is our
	# y data
	# We also need to bootstrap our errors in x, but this has uniform errors so we
	# can use the standard Gaussian method of choosing our new data point
	#
	#
	#
	#
	#
	# We now have our new x data points, so we can run the fit on the data
	# Remember, our y errors are non-uniform so we have to use the average error
	# How do you deal with upper limits?
	#
	#
	#
	#
	# Save the results of this run, try again


			

# Remember, non-uniform errors so have to average errors - mirror errors instead



# How do you deal with upper limits?




















