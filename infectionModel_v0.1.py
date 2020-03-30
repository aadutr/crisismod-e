# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
import os

# save the figures in a separate folder
if not os.path.exists('figures'):
  os.mkdir('figures')

colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']

def populationModel(t, n, r, p):
  """ Define the population model based on the different rates and initial conditions 

    Parameters
    ----------
    t : Time array
    n : Fraction of the population that is in a given state (susceptible, asymptomatic, symptomatic, hospitalized, recovered or dead).
    r : Rates array.
    p : Defines the capacity of the health care system (currently determines how many IC beds there are for the population).

    Returns
    ------
    Array dn containing the combination of rates that describe the system of ODEs 
    
    """
    
  HP  = n[0] #susceptible people
  IP1 = n[1] #infected but not symptomatic people
  IP2 = n[2] #symptomatic patients
  IP3 = n[3] #hospitalized patients
  RP  = n[4] #recovered people
  DP  = n[5] #dead people

  TP = HP + IP1 + IP2 + IP3 + RP #total population

  r_meeting1      = r[ 0] #rate at which susceptible people meet asymptomatic patients
  r_meeting2      = r[ 1] #rate at which susceptible people meet symptomatic patients
  r_meeting3      = r[ 2] #rate at which susceptible people meet hospitalized patients
  r_infection1    = r[ 3] #rate at which asymptomatic patients infect people
  r_infection2    = r[ 4] #rate at which symptomatic patients infect people
  r_infection3    = r[ 5] #rate at which hospitalized patients infect people
  r_sym           = r[ 6] #rate at which asymptomatic patients get symptoms
  r_hos           = r[ 7] #rate at which symptomatic patients become hospitalized
  r_d1            = r[ 8] #death rate 1: rate constant for dying due to the disease
  r_d2            = r[ 9] #death rate 2: relative increae in death rate when IC beds are full 
  r_im1           = r[10] #recovery rate for asymptomatic patients
  r_im2           = r[11] #recovery rate for symptomatic patients
  r_im3           = r[12] #recovery rate for hospitalized patients
  
  cap_IC          = p[ 0] #number of IC beds

  R_inf1 = r_meeting1 * r_infection1 * (HP / TP) * IP1 #infection rate 1: chance they susceptible people meet asymptomatic patients ánd that they are infected
  R_inf2 = r_meeting2 * r_infection2 * (HP / TP) * IP2 #infection rate 2: people are infected by symptomatic patients
  R_inf3 = r_meeting3 * r_infection3 * (HP / TP) * IP3 #infection rate 3: people are infected by hospitalized patients
  r_d = r_d1 + logistic(IP3, cap_IC) * r_d2 * r_d1
  
  dn = np.empty(len(n)) #create an empty array to define the ODEs

  dn[0] = - R_inf1 - R_inf2 - R_inf3
  dn[1] = + R_inf1 + R_inf2 + R_inf3 - r_sym * IP1               - r_im1 * IP1
  dn[2] =                            + r_sym * IP1 - r_hos * IP2               - r_im2 * IP2  
  dn[3] =                                          + r_hos * IP2                             - r_im3 * IP3 - r_d * IP3
  dn[4] =                                                        + r_im1 * IP1 + r_im2 * IP2 + r_im3 * IP3
  dn[5] =                                                                                                  + r_d * IP3
  
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
    
# r_meeting1  r_meeting2  r_meeting3  r_infection1  r_infection2  r_infection3  r_sym  r_hos   r_d1  r_d2  r_im1  r_im2  r_im3
r = [     10,          5,          1,         0.01,          0.3,          0.1,  0.05,  0.02,  0.01, 0.04,  0.05,  0.08,   0.1]

p = [1250 / 8.4e6] # Determines how many IC beds are available. Now 1250 beds for 8.4 mln people
n0 = [0.9999975, 0.0000025, 0, 0, 0, 0]  #initial conditions: fraction of the population that is in a certain state. 

#creating the time array
tstart = 0
tend   = 365
tspan = (tstart, tend)
t = np.linspace(tstart, tend, tend * 100)

#solving the system of ODEs 
sol = solve_ivp(populationModel, tspan, n0, args=[r, p], dense_output=True)
print(sol.message)
y = sol.sol(t)

#plotting the solution
fig1, ax1 = plt.subplots()

ax1.plot(t, y.transpose(), linewidth=3)
ax1.legend(['HP', 'IP1', 'IP2', 'IP3', 'RP', 'DP'])
ax1.set_title('Population disease model')
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')

fig1.savefig('figures/Population disease model_1.png')

plt.show()
