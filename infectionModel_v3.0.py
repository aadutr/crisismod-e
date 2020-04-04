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
country = 'sicilia' #pick 'iceland' for Iceland
datafile = country + '_data.csv'
parameterfile = country + '_parameters.txt'


#Declaration of the input variables

input_dict = file_to_dict(parameterfile)    
vals = np.fromiter(input_dict.values(), dtype=float)

#sorry for this mess feel free to make a function of this
params = Parameters() #special type of parameters as defined by the lmfit module
params.add('r_meeting1', value=vals[0], vary=True, min=0)
params.add('r_meeting2', value=vals[1], vary=True,min=0)
params.add('r_meeting3', value=vals[2], vary=True,min=0)
params.add('r_meeting4', value=vals[3], vary=True,min=0)
params.add('r_infection1', value=vals[4], vary=True,min=0)
params.add('r_infection2', value=vals[5], vary=True,min=0)
params.add('r_infection3', value=vals[6], vary=True,min=0)
params.add('r_infection4', value=vals[7], vary=True,min=0)
params.add('r_sym', value=vals[8], vary=True,min=0)
params.add('r_hos', value=vals[9], vary=True,min=0)
params.add('r_d1', value=vals[10], vary=True,min=0)
params.add('r_d2', value=vals[11], vary=True,min=0)
params.add('r_im1', value=vals[12], vary=True,min=0)
params.add('r_im2', value=vals[13], vary=True,min=0)
params.add('r_im3', value=vals[14], vary=True,min=0)
params.add('r_ic', value=vals[15], vary=True,min=0)
params.add('r_rehos', value=vals[16], vary=True,min=0)
params.add('n0_susc', value=vals[17], vary=False)
params.add('n0_inf1', value=vals[18], vary=False)
params.add('n0_inf2', value=vals[19], vary=False)
params.add('n0_inf3', value=vals[20], vary=False)
params.add('n0_inf4', value=vals[21], vary=False)
params.add('n0_rec', value=vals[22], vary=False)
params.add('n0_dead', value=vals[23], vary=False)
params.add('n_beds', value=vals[24], vary=False)
params.add('pop_size', value=vals[25], vary=False)

n0 = [params['n0_susc'].value, params['n0_inf1'].value, params['n0_inf2'].value, params['n0_inf3'].value, params['n0_inf4'].value, params['n0_rec'].value, params['n0_dead'].value]

#loading the actual data
country_data = data_loader(datafile,params['pop_size'].value)
country_data = np.transpose(country_data)

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

# making a time array for the model fitting (this is smaller than 365 days bc we have only 8 datapoints)
amount_of_days = country_data.shape[0]
t_measured = np.linspace(0, amount_of_days - 1, amount_of_days)
# cat2_measured = country_data[:,1]

# fit model
result = minimize(residual, params, args=(t_measured, country_data), method='leastsq',nan_policy='raise')  # leastsq nelder
# check results of the fit
data_fitted = g(np.linspace(0., 32., 100), n0, result.params)

# plot fitted data
fig1, ax1 = plt.subplots()
for i in range(0,np.size(country_data,1)):
    ax1.scatter(t_measured, country_data[:,i], marker='o', color=colors[i], label='measured data', s=75)

ax1.plot(np.linspace(0., 32., 100), data_fitted)
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "upper left")
ax1.set_title('Model fit for ' + country)
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')
ax1.set_ylim([0, 1.1 * max(data_fitted[:, 1])])

# display fitted statistics
report_fit(result)

plt.show()
fig1.savefig('figures/Model_fit_'+ country +'.png')

#run the ODE model with the new parameters for the entire time array and plot results
new_params = result.params
data_fitted2 = g(t, n0, result.params)

fig2, ax2 = plt.subplots()
ax2.plot(t,data_fitted2)
ax2.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "upper right")
ax2.set_title('Population disease model with fitted rates')
ax2.set_ylabel('Fraction of people')
ax2.set_xlabel('Time (days)')

fig2.savefig('figures/Population disease model '+ country +'.png')


#solving the system of ODEs 
# sol = solve_ivp(populationModel, n0, tspan, args=[new_params], dense_output=True)
# print(sol.message)
# y = sol.sol(t)


# #plotting the solution
# fig1, ax1 = plt.subplots()

# ax1.plot(t, y.transpose(), linewidth=3)
# ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "right")
# ax1.set_title('Population disease model')
# ax1.set_ylabel('Fraction of people')
# ax1.set_xlabel('Time (days)')

# fig1.savefig('figures/Population disease model_2.png')

# #calculating the per day increase of cases
# # The idea is if you have an array [1,2,3] and you want to know the increments between
# # each element you can create an array [0, 1, 2] and subtract that from the original matrix
# # and then you will get the differences between each consecutive element.
# y_shift = np.zeros((len(y)-1,tend))     #empty matrix
# endpoint = tend*100
# y_perday = y[:,0:endpoint:100]          #Taking out the data points for each exact day
# y_shift[:,1:] = y_perday[1:,0:tend-1:1] #Putting it in the new matrix with position shifted by 1
# delta_y = (y_perday[1:,:] - y_shift)    #calculating the difference in cases for each category between consecutive days
# t_days = np.linspace(tstart, tend, tend)#having a correct time array for plotting purposes

# model_mse = mse_calculator(y_perday,iceland_data)
# #rates_opt = minimize(mse_calculator(y_perday,iceland_data),r0)
# #rates_opt = curve_fit

# #calculating the error between 
# #plotting the amount of new cases per day per group
# fig2, ax2 = plt.subplots()
# ax2.plot(t_days,delta_y.transpose(), linewidth = 3)
# ax2.legend([ 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "right")
# ax2.set_title('New cases/day of disease')
# ax2.set_ylabel('New cases as fraction of people')
# ax2.set_xlabel('Time (days)')

# fig2.savefig('figures/newcaserate.png')

# plt.show()




