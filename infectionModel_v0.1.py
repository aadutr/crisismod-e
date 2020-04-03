# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
import os
from function_lib import *

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')

colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']


#Declaration of the input variables
params = file_to_dict("parameters_new.txt")                        
n0 = [params["n0_susc"], params["n0_inf1"], params["n0_inf2"],
      params["n0_inf3"], params["n0_inf4"], params["n0_rec"], params["n0_dead"]]  #initial conditions: fraction of the population that is in a certain state. 

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

#solving the system of ODEs 
sol = solve_ivp(populationModel, tspan, n0, args=[ params], dense_output=True)
print(sol.message)
y = sol.sol(t)

#loading the actual data
iceland_data = data_loader('iceland_data.csv',params["pop_size"])
 
#plotting the solution
fig1, ax1 = plt.subplots()

ax1.plot(t, y.transpose(), linewidth=3)
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "right")
ax1.set_title('Population disease model')
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')

fig1.savefig('figures/Population disease model_2.png')

#calculating the per day increase of cases
# The idea is if you have an array [1,2,3] and you want to know the increments between
# each element you can create an array [0, 1, 2] and subtract that from the original matrix
# and then you will get the differences between each consecutive element.
y_shift = np.zeros((len(y)-1,tend))     #empty matrix
endpoint = tend*100
y_perday = y[:,0:endpoint:100]          #Taking out the data points for each exact day
y_shift[:,1:] = y_perday[1:,0:tend-1:1] #Putting it in the new matrix with position shifted by 1
delta_y = (y_perday[1:,:] - y_shift)    #calculating the difference in cases for each category between consecutive days
t_days = np.linspace(tstart, tend, tend)#having a correct time array for plotting purposes

#plotting the amount of new cases per day per group
fig2, ax2 = plt.subplots()
ax2.plot(t_days,delta_y.transpose(), linewidth = 3)
ax2.legend([ 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "right")
ax2.set_title('New cases/day of disease')
ax2.set_ylabel('New cases as fraction of people')
ax2.set_xlabel('Time (days)')

fig2.savefig('figures/newcaserate.png')

plt.show()



