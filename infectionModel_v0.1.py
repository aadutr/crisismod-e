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
params = file_to_dict("parameters.txt")                        
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
 
#plotting the solution
fig1, ax1 = plt.subplots()

ax1.plot(t, y.transpose(), linewidth=3)
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'ICU', 'Recovered', 'Dead'], loc = "right")
ax1.set_title('Population disease model')
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')

fig1.savefig('figures/Population disease model_2.png')

plt.show()


