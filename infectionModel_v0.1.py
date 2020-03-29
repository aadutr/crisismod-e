

# disease spread
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
import os

# make room for figures
if not os.path.exists('figures'):
  os.mkdir('figures')

colors = ['#299727ff', '#b6311cff', '#276ba2ff', '#424242ff']

def populationModel(t, n, r, p):
  HP  = n[0]
  SP1 = n[1]
  SP2 = n[2]
  SP3 = n[3]
  IP  = n[4]
  DP  = n[5]

  TP = HP + SP1 + SP2 + SP3 + IP

  r_meeting1      = r[ 0]
  r_meeting2      = r[ 1]
  r_meeting3      = r[ 2]
  r_infection1    = r[ 3]
  r_infection2    = r[ 4]
  r_infection3    = r[ 5]
  r_sym           = r[ 6]
  r_hos           = r[ 7]
  r_d1            = r[ 8]
  r_d2            = r[ 9]
  r_im1           = r[10]
  r_im2           = r[11]
  r_im3           = r[12]
  
  cap_ER          = p[ 0]

  R_inf1 = r_meeting1 * r_infection1 * (HP / TP) * SP1
  R_inf2 = r_meeting2 * r_infection2 * (HP / TP) * SP2
  R_inf3 = r_meeting3 * r_infection3 * (HP / TP) * SP3
  r_d = r_d1 + logistic(SP3, cap_ER) * r_d2
  
  dn = np.empty(len(n))

  dn[0] = - R_inf1 - R_inf2 - R_inf3
  dn[1] = + R_inf1 + R_inf2 + R_inf3 - r_sym * SP1               - r_im1 * SP1
  dn[2] =                            + r_sym * SP1 - r_hos * SP2               - r_im2 * SP2  
  dn[3] =                                          + r_hos * SP2                             - r_im3 * SP3 - r_d * SP3
  dn[4] =                                                        + r_im1 * SP1 + r_im2 * SP2 + r_im3 * SP3
  dn[5] =                                                                                                  + r_d * SP3
  
  return dn

# adapted from https://math.stackexchange.com/a/2869151
def jacPopulationModel(t, n, r):
  eps = np.finfo(np.float).eps
  J = np.zeros([len(n), len(n)])

  for i in range(len(n)):
    n1 = n.copy()
    n2 = n.copy()

    n1[i] += eps
    n2[i] -= eps

    f1 = populationModel(t, n1, r)
    f2 = populationModel(t, n2, r)

    J[:, i] = (f1 - f2) / (2 * eps)

  return J

def logistic(SP3, cap_ER):
  return np.exp(SP3 - cap_ER) / (np.exp(SP3 - cap_ER) + 1)

# r_meeting1  r_meeting2  r_meeting3  r_infection1  r_infection2  r_infection3  r_sym  r_hos   r_d1  r_d2  r_im1  r_im2  r_im3
r = [     10,          5,          1,         0.01,          0.3,          0.1,  0.05,  0.02,  0.01, 0.04,  0.05,  0.08,   0.1]

p = [1250 / 8.4e6] # 1250 beds for 8.4 mln people

tstart = 0
tend   = 365
tspan = (tstart, tend)
# print(tspan.shape())
n0 = [0.9999975, 0.0000025, 0, 0, 0, 0]

sol = solve_ivp(populationModel, tspan, n0, args=[r, p], dense_output=True)

print(sol.message)

t = np.linspace(tstart, tend, tend * 100)

y = sol.sol(t)

fig1, ax1 = plt.subplots()

# ax1.set_prop_cycle(cycler(color=colors))

ax1.plot(t, y.transpose(), linewidth=3)
ax1.legend(['HP', 'SP1', 'SP2', 'SP3', 'IP', 'DP'])
ax1.set_title('Population disease model')
ax1.set_ylabel('Fraction of people')
ax1.set_xlabel('Time (days)')

fig1.savefig('figures/Population disease model_1.png')

plt.show()
