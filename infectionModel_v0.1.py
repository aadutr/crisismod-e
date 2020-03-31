# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
import os

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')

colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']

def populationModel(t, n, p, params):
  """ Define the population model based on the different rates and initial conditions 

    Parameters
    ----------
    t : Time array
    n : Fraction of the population that is in a given state (susceptible, asymptomatic, symptomatic, hospitalized, recovered or dead).
    r : params array.
    p : Defines the capacity of the health care system (currently determines how many IC beds there are for the population).

    Returns
    ------
    Array dn containing the combination of rates that describe the system of ODEs 
    
    """  
  SP  = n[0]
  IP1 = n[1]
  IP2 = n[2]
  IP3 = n[3]
  RP  = n[4]
  DP  = n[5]

  TP = SP + IP1 + IP2 + IP3 + RP

  #TP = params["n0_susc"] + params["n0_inf1"] + params["n0_inf2"] + params["n0_inf3"] + params["n0_rec"] #total population
 
  R_inf1 = params["r_meeting1"] * params["r_infection1"] * (SP / TP) * IP1 #infection rate 1: chance they susceptible people meet asymptomatic patients ánd that they are infected
  R_inf2 = params["r_meeting2"] * params["r_infection2"] * (SP / TP) * IP2 #infection rate 2: people are infected by symptomatic patients
  R_inf3 = params["r_meeting3"] * params["r_infection3"] * (SP / TP) * IP3 #infection rate 3: people are infected by hospitalized patients
  r_d = params["r_d1"] + logistic(IP3, cap_IC) * params["r_d2"] * params["r_d1"]
  
  dn = np.empty(len(n)) #create an empty array to define the ODEs

  dn[0] = - R_inf1 - R_inf2 - R_inf3
  dn[1] = + R_inf1 + R_inf2 + R_inf3 - params["r_sym"] * IP1 - params["r_im1"] * IP1
  dn[2] = + params["r_sym"] * IP1 - params["r_hos"] * IP2 - params["r_im2"] * IP2  
  dn[3] = + params["r_hos"] * IP2 - params["r_im3"] *IP3 - r_d * IP3
  dn[4] = + params["r_im1"] * IP1 + params["r_im2"] * IP2 + params["r_im3"]* IP3
  dn[5] = + r_d * IP3
  
  return dn


def logistic(IP3, cap_IC):
    """ Creates a sigmoid shape that accomodates for the increase in death rate when the IC beds are full. You need this because otherwise the ODEs can not be solved. 

    Parameters
    ----------
    IP3 : Float
    Fraction of the population that is hospitalized.
    cap_IC : Float
    Maximum capacity of the IC.

    Returns
    -------
    FLOAT
    Sigmoid function.

    """
    return np.exp(IP3 - cap_IC) / (np.exp(IP3 - cap_IC) + 1)


#Declaration of the input variables

params = {}

with open("parameters.txt") as f:
    for line in f:
        if not line.startswith("#"):
            (key,val) = line.split()
            params[key] = float(val)
                        
cap_IC = params["n_beds"] / params["pop_size"] # Determines how many IC beds are available.
n0 = [params["n0_susc"], params["n0_inf1"], params["n0_inf2"], params["n0_inf3"], params["n0_rec"], params["n0_dead"]]  #initial conditions: fraction of the population that is in a certain state. 

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

#solving the system of ODEs 
sol = solve_ivp(populationModel, tspan, n0, args=[cap_IC, params], dense_output=True)
print(sol.message)
y = sol.sol(t)

#plotting the solution
fig1, ax1 = plt.subplots()

ax1.plot(t, y.transpose(), linewidth=3)
ax1.legend(['Susceptible', 'Asymptomatic', 'Symptomatic', 'Hospitalized', 'Recovered', 'Dead'])
ax1.set_title('Population disease model')
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')

fig1.savefig('figures/Population disease model_1.png')

plt.show()


