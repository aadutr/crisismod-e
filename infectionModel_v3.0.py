# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import os
from function_lib_new import *
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report, conf_interval, Minimizer
from scipy.integrate import odeint
import pickle

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')


# colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']
colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y']

# what country/region?
country = 'sicilia' #pick 'iceland' for Iceland
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

#Below is construction to use when running the minimizer on multiple datasets
#and saving the parameters to compare later on.
#saving the minimizer result to a temporary file for later usage
with open('par.pkl', 'wb') as f:
    pickle.dump(result, f)
#loading in the minimizer result again   
with open('./par.pkl', 'rb') as f:
    list2 = pickle.load(f)

# plot fitted data
fig1, ax1 = plt.subplots()
for i in range(0,np.size(country_data,1)):
    ax1.scatter(t_measured, country_data[:,i], marker='o', color=colors[i], label='measured data', s=30)
    ax1.plot(np.linspace(0., amount_of_days, 100), data_fitted[:,i], color = colors[i])
    
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "best")
plt.suptitle('Model fit for ' + country, fontsize = 14)
subtitle_string = 'Weights used: Susc. ' + str(params['w_susc'].value) + ', Asymp. ' + str(params['w_asym'].value) + ', Symp. ' + str(params['w_sym'].value)+ ', Hosp. ' + str(params['w_hos'].value)+ ', ICU ' + str(params['w_icu'].value)+ ', Rec. ' + str(params['w_rec'].value)+ ', Dead ' + str(params['w_dead'].value)
plt.title(subtitle_string, fontsize = 8)
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')
#ax1.set_ylim([0,0.00020])
ax1.set_ylim([0, 1.1 * max(data_fitted[:, 2])])

plt.show()
fig1.savefig('figures/Model_fit_'+ country +'.png')

#run the ODE model with the new parameters for the entire time array and plot results
new_params = result.params
data_fitted2 = g(t, n0, result.params)

fig2, ax2 = plt.subplots()
for i in range(0,np.size(country_data,1)):
    ax2.plot(t,data_fitted2[:,i],color=colors[i])

ax2.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "upper right")
plt.suptitle('Population disease model for ' + country, fontsize = 14)
subtitle_string = 'Weights used: Susc. ' + str(params['w_susc'].value) + ', Asymp. ' + str(params['w_asym'].value) + ', Symp. ' + str(params['w_sym'].value)+ ', Hosp. ' + str(params['w_hos'].value)+ ', ICU ' + str(params['w_icu'].value)+ ', Rec. ' + str(params['w_rec'].value)+ ', Dead ' + str(params['w_dead'].value)
plt.title(subtitle_string, fontsize = 8)
ax2.set_ylabel('Fraction of people')
ax2.set_xlabel('Time (days)')

fig2.savefig('figures/Population disease model '+ country +'.png')


