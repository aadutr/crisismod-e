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

def populationModel(t, n, r, pop_info):
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
  IP4 = n[4]
  RP  = n[5]
  DP  = n[6]
  

  TP = SP + IP1 + IP2 + IP3 + IP4 + RP
  cap_IC = pop_info[0] / pop_info[1] # Determines how many IC beds are available.

  R_inf1 = r[0] * r[4] * (SP / TP) * IP1 #infection rate 1: chance they susceptible people meet asymptomatic patients ánd that they are infected
  R_inf2 = r[1] * r[5] * (SP / TP) * IP2 #infection rate 2: people are infected by symptomatic patients
  R_inf3 = r[2] * r[6] * (SP / TP) * (IP3) #infection rate 3: people are infected by hospitalized patients
  R_inf4 = r[3] * r[7] * (SP / TP) * (IP4) #infection rate 3: people are infected by ICU patients
  r_d = r[10] + logistic(IP4, cap_IC) * r[11] * r[10]
  
  #R_inf1 = params["r_meeting1"] * params["r_infection1"] * (SP / TP) * IP1 #infection rate 1: chance they susceptible people meet asymptomatic patients ánd that they are infected
  #R_inf2 = params["r_meeting2"] * params["r_infection2"] * (SP / TP) * IP2 #infection rate 2: people are infected by symptomatic patients
  #R_inf3 = params["r_meeting3"] * params["r_infection3"] * (SP / TP) * (IP3) #infection rate 3: people are infected by hospitalized patients
  #R_inf4 = params["r_meeting4"] * params["r_infection4"] * (SP / TP) * (IP4) #infection rate 3: people are infected by ICU patients
  #r_d = params["r_d1"] + logistic(IP4, cap_IC) * params["r_d2"] * params["r_d1"]
  
  dn = np.empty(len(n)) #create an empty array to define the ODEs
  
  # v2.3
  dn[0] = - R_inf1 - R_inf2 - R_inf3 - R_inf4
  dn[1] = + R_inf1 + R_inf2 + R_inf3 + R_inf4 - r[8] * IP1 - r[12] * IP1
  dn[2] = + r[8] * IP1 - r[9] * IP2 - r[13] * IP2 - r_d * IP2 
  dn[3] = + r[9] * IP2 - r[14] * IP3 - r_d * IP3 + r[16] * IP4 - r[15] * IP3
  dn[4] = + r[15] * IP3 - r[16] * IP4 -r_d * IP4;
  dn[5] = + r[12] * IP1 + r[13] * IP2 + r[14]* IP3
  dn[6] = + r_d * (IP2 + IP3 + IP4)
  
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


    data = np.delete(data,[0,1,7],axis=1) #delete country, date and tests
    data = np.delete(data,[0],axis=0) #delete header
    data = np.transpose(data)
    data = data.astype(np.float)
    data = np.divide(data,pop_size)
    sum_categories = np.sum(data,axis=0)
    sum_categories = np.reshape(sum_categories,(1,8))
    sum_categories = 1-sum_categories
    
    data = np.concatenate((sum_categories,data),axis=0)
  
    return(data)

def mse_calculator(model,data):
    """
    
    Parameters
    ----------
    model : Array of float
        fractions of the population in a category at each day as calculated by the model
    data : Array of float
        fractions of the population in a categorty at each day as observed in real-life 

    Returns
    -------
    mse_array: Array of float
        mse for each day

    """
    
    model_compare = np.delete(model,slice(np.size(data,1),np.size(model,1)),axis=1) #select only the days that you have data for to compare
    mse = (np.square(model_compare - data)).mean(axis=0) #calculate the mse between the model and the data at each day
    
    return(mse)
