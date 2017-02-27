#!/usr/bin/python

import numpy as np;

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    #print predictions;
    #print ages;
    #print net_worths;
    
    for i, net in enumerate(net_worths):
        cleaned_data.append((int(ages[i]), float(net), np.abs(float(net - predictions[i]))));
    cleaned_data.sort(key=lambda item: item[2]);
    return cleaned_data[:int(len(cleaned_data) * 0.9)];

