# -*- coding: utf-8 -*-
"""
Function library
"""

# importing required modules
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import solve_ivp
import os
import csv
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

def populationModel(n, t, paras):
  """ Define the population model based on the different rates and initial conditions 

    Parameters
    ----------
    t : Time array
    n : Fraction of the population that is in a given state (susceptible, asymptomatic, symptomatic, hospitalized, recovered or dead).
    paras: input parameters (initial conditions, initial rates and state of the healthcare system)
    Returns
    ------
    Array dn containing the combination of rates that describe the system of ODEs 
    
    """  
  SP  = n[0]
  IP1 = n[1]
  IP2 = n[2]
  IP3 = n[3]
  IP4 = n[4]
  RP  = n[5]
  DP  = n[6]
  

  TP = SP + IP1 + IP2 + IP3 + IP4 + RP
  cap_IC = paras['n_beds'].value / paras['pop_size'].value # Determines how many IC beds are available.
  
  #rates are calculated using an inverse sigmoid from r_meeting1 (before measures) to r_meeting1b (after measures)
  R_inf1 = rate_sigmoid(paras['r_meeting1'].value, paras['r_meeting1b'].value, t, paras['measures_time'].value) * paras['r_infection1'].value * (SP / TP) * IP1 #infection rate 1: chance they susceptible people meet asymptomatic patients ánd that they are infected
  R_inf2 = rate_sigmoid(paras['r_meeting2'].value, paras['r_meeting2b'].value, t, paras['measures_time'].value) * paras['r_infection2'].value * (SP / TP) * IP2 #infection rate 2: people are infected by symptomatic patients
  R_inf3 = rate_sigmoid(paras['r_meeting3'].value, paras['r_meeting3b'].value, t, paras['measures_time'].value) * paras['r_infection3'].value * (SP / TP) * (IP3) #infection rate 3: people are infected by hospitalized patients
  R_inf4 = rate_sigmoid(paras['r_meeting4'].value, paras['r_meeting4b'].value, t, paras['measures_time'].value) * paras['r_infection4'].value * (SP / TP) * (IP4) #infection rate 3: people are infected by ICU patients
  r_dicu = paras['r_d1'].value + logistic(IP4, cap_IC) * paras['r_d1'] * paras['r_d2']
  r_dhos = paras['r_d0']
  dn = np.empty(len(n)) #create an empty array to define the ODEs
    
   # v2.3
  dn[0] = - R_inf1 - R_inf2 - R_inf3 - R_inf4
  dn[1] = + R_inf1 + R_inf2 + R_inf3 + R_inf4 - paras['r_sym'].value * IP1 - paras['r_im1'].value * IP1
  dn[2] = + paras['r_sym'].value * IP1 - paras['r_hos'].value * IP2 - paras['r_im2'].value * IP2 #- r_d * IP2 
  dn[3] = + paras['r_hos'].value * IP2 - paras['r_im3'].value * IP3 - r_dhos * IP3 + paras['r_rehos'].value * IP4 - paras['r_ic'].value * IP3
  dn[4] = + paras['r_ic'].value * IP3 - paras['r_rehos'].value * IP4 -r_dicu * IP4;
  dn[5] = + paras['r_im1'].value * IP1 + paras['r_im2'].value * IP2 + paras['r_im3'].value* IP3
  dn[6] = + r_dhos * IP3 + r_dicu* IP4
  
  return dn

def g(t, n0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    #this is a different ODE solver than we used before (that was solve_ivp), so we might want to check the differences and see if that matters
    n = odeint(populationModel, n0, t, args=(paras,))
    return n

def residual(paras, t, data):

    """
    compute the residual between actual data and fitted data
    """

    n0 = paras['n0_susc'].value, paras['n0_inf1'].value, paras['n0_inf2'].value, paras['n0_inf3'].value, paras['n0_inf4'].value, paras['n0_rec'].value, paras['n0_dead'].value
    model = g(t, n0, paras)
    residual = model-data
    #give hospitalized, ICU & dead people a higher weight
    for i in [3,4,6]:
        residual[:,i]=10*residual[:,i]
    
    return residual.ravel()

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

def rate_sigmoid(r_meeting,r_meetingb,t,measures_time):
    """
    

    Parameters
    ----------
    r_meeting : FLOAT
        Rate at which people meet BEFORE measures.
    r_meetingb : FLOAT
        Rate at which people meet AFTER measures.
    t : INT
        Time.
    measures_time : FLOAT
        Time in days at which the measures were implemented.

    Returns
    -------
    None.

    """
    return((r_meeting - r_meetingb)*(1-(1/(1+np.exp(-(t-measures_time)))))+ r_meetingb)

def file_to_dict(filename):
    params = {}

    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                (key,val) = line.split()
                params[key] = float(val)
                
    return params

def data_loader(filename,pop_size):
    """
    Load the actual data in an array so that we can fit the model to it

    Parameters
    ----------
    filename : string
        CSV file containing the country, date, number of people in each category and number of tests.

    Returns
    -------
    array with fraction of population in each category

    """
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile))
   
        data = np.array(data)


    data = np.delete(data,[0,1,7,9],axis=1) #delete country, date and tests and comments
    data = np.delete(data,[0],axis=0) #delete header
    data = np.transpose(data)
    data = data.astype(np.float)
    data = np.divide(data,pop_size)
    sum_categories = np.sum(data,axis=0)
    sum_categories = np.reshape(sum_categories,(1,data.shape[1]))
    sum_categories = 1-sum_categories
    
    data = np.concatenate((sum_categories,data),axis=0)
  
    return(data)

def parameters(input_dict,country_data):
    params = Parameters() #special type of parameters as defined by the lmfit module
    params.add('r_meeting1', value=input_dict["r_meeting1"], vary=True, min=0)
    params.add('r_meeting2', value=input_dict["r_meeting2"], vary=True, min=0)
    params.add('r_meeting3', value=input_dict["r_meeting3"], vary=True, min=0)
    params.add('r_meeting4', value=input_dict["r_meeting4"], vary=True,min=0)
    params.add('r_meeting1b', value=input_dict["r_meeting1b"], vary=True, min=0)
    params.add('r_meeting2b', value=input_dict["r_meeting2b"], vary=True, min=0)
    params.add('r_meeting3b', value=input_dict["r_meeting3b"], vary=True, min=0)
    params.add('r_meeting4b', value=input_dict["r_meeting4b"], vary=True, min=0)
    params.add('r_infection1', value=input_dict["r_infection1"], vary=True,min=0)
    params.add('r_infection2', value=input_dict["r_infection2"], vary=True,min=0)
    params.add('r_infection3', value=input_dict["r_infection3"], vary=True,min=0)
    params.add('r_infection4', value=input_dict["r_infection4"], vary=True,min=0)
    params.add('r_sym', value=input_dict["r_sym"], vary=True,min=0)
    params.add('r_hos', value=input_dict["r_hos"], vary=True,min=0)
    params.add('r_d0', value=input_dict["r_d0"], vary=True,min=0)
    params.add('r_d1', value=input_dict["r_d1"], vary=True,min=0)
    params.add('r_d2', value=input_dict["r_d2"], vary=False,min=0)
    params.add('r_im1', value=input_dict["r_im1"], vary=True,min=0)
    params.add('r_im2', value=input_dict["r_im2"], vary=True,min=0)
    params.add('r_im3', value=input_dict["r_im3"], vary=True,min=0)
    params.add('r_ic', value=input_dict["r_ic"], vary=True,min=0)
    params.add('r_rehos', value=input_dict["r_rehos"], vary=True,min=0)
    params.add('n0_susc', value=country_data[0,0], vary=False)
    params.add('n0_inf1', value=country_data[0,1], vary=False)
    params.add('n0_inf2', value=country_data[0,2], vary=False)
    params.add('n0_inf3', value=country_data[0,3], vary=False)
    params.add('n0_inf4', value=country_data[0,4], vary=False)
    params.add('n0_rec', value=country_data[0,5], vary=False)
    params.add('n0_dead', value=country_data[0,6], vary=False)
    params.add('n_beds', value=input_dict["n_beds"], vary=False)
    params.add('pop_size', value=input_dict["pop_size"], vary=False)
    params.add('measures_time', value=input_dict["measures_time"], vary=False)

    return params
