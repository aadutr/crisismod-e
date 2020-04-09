# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import os
from function_lib_new import *
from lmfit import minimize, Parameters, Parameter, report_fit, fit_report
from scipy.integrate import odeint

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')


# colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']
colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y']

# what country/region?
countries = ['sicilia', 'campania', 'iceland', 'geneve'] #pick 'iceland' for Iceland

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

#create storage for fits
fits = []

for country in countries:
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

    # making a time array for the model fitting using the available data
    amount_of_days = country_data.shape[0]
    t_measured = np.linspace(0, amount_of_days - 1, amount_of_days)

    # fit model
    result = minimize(residual, params, args=(t_measured, country_data), method='leastsq',nan_policy='raise')  # leastsq nelder
    # check results of the fit
    data_fitted = g(np.linspace(0., amount_of_days, 100), n0, result.params)
    #saving the minimizer result to a temporary file for later usage
    fits.append(result)
    #check to see where it is
    print("fitted " + country)
    
# display fitted statistics
#report_fit(result)
i = 0
fitted_params = np.zeros((len(params),len(countries)))

for minresult in fits:
    fitted_params[:,i] = list(minresult.params.values())
    i+=1
    
# n = 0 
# for name, val in fits[0].params.items():
#     toprint = name + str(fitted_params[n,:])
#     print(toprint)
#     n+=1
    
#the parameters we expect to be the same are everything but r_meetings, so for
#use in further models we will set these paramters to the median of the found ones
to_be_deleted = list(range(0,8,1)) + list(range(22,39,1))
stable_params = np.delete(fitted_params, to_be_deleted, axis=0) #delete the rows of the params that should differ per set
stable_params = np.median(stable_params, axis=1)
newline = '#\n'
to_write = ''
keys = list(fits[0].params.keys())
for index in range(8,22,1):
    to_write+=newline
    to_write+=keys[index] + ' ' + str(stable_params[index-8]) + '\n'
    

with open("stable_parameters.txt", "w") as file:
    file.write(to_write)