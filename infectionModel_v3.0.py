# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import os
from function_lib_new import *
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')


# colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']
colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y']

# what country/region?
country = 'campania' #pick 'iceland' for Iceland
datafile = country + '_data.csv'
parameterfile = country + '_parameters.txt'
input_dict = file_to_dict(parameterfile)  

#loading the actual data
country_data = data_loader(datafile,input_dict["pop_size"])
country_data = np.transpose(country_data)


#Declaration of the input variables 
vals = np.fromiter(input_dict.values(), dtype=float) #get values from dictionary
params = parameters(input_dict,country_data) #define parameters, some are fixed and some are variable (see function_lib_new)
n0 = [params['n0_susc'].value, params['n0_inf1'].value, params['n0_inf2'].value, #initial conditions
      params['n0_inf3'].value, params['n0_inf4'].value, params['n0_rec'].value, 
      params['n0_dead'].value]

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

# making a time array for the model fitting using the available data
amount_of_days = country_data.shape[0]
t_measured = np.linspace(0, amount_of_days - 1, amount_of_days)

# fit model
result = minimize(residual, params, args=(t_measured, country_data), method='leastsq',nan_policy='raise')  # leastsq nelder
# check results of the fit
data_fitted = g(np.linspace(0., amount_of_days, 100), n0, result.params)
# display fitted statistics
report_fit(result)

# plot fitted data
fig1, ax1 = plt.subplots()
for i in range(0,np.size(country_data,1)):
    ax1.scatter(t_measured, country_data[:,i], marker='o', color=colors[i], label='measured data', s=30)
    ax1.plot(np.linspace(0., amount_of_days, 100), data_fitted[:,i], color = colors[i])
    
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "upper left")
ax1.set_title('Model fit for ' + country)
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')
ax1.set_ylim([0,0.00020])
#ax1.set_ylim([0, 1.1 * max(data_fitted[:, 1])])

plt.show()
fig1.savefig('figures/Model_fit_'+ country +'.png')

#run the ODE model with the new parameters for the entire time array and plot results
new_params = result.params
data_fitted2 = g(t, n0, result.params)

fig2, ax2 = plt.subplots()
for i in range(0,np.size(country_data,1)):
    ax2.plot(t,data_fitted2[:,i],color=colors[i])

ax2.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "upper right")
ax2.set_title('Population disease model with fitted rates')
ax2.set_ylabel('Fraction of people')
ax2.set_xlabel('Time (days)')

fig2.savefig('figures/Population disease model '+ country +'.png')


