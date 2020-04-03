# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:20:27 2020

@author: myrth
"""

import csv
import numpy as np

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

pop_size = 364260
iceland_data = data_loader('iceland_data.csv',pop_size)