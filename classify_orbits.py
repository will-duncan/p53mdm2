from ode_functions import make_parameters, fun, plot_time_series, plot_projections, ic_function, convert_to_dict_2, classify_orbit, theta_from_param
from periodic_orbits import get_periodic_domains
import scipy.integrate
import numpy as np
import pandas as pd
import random
import json
import time

def IC_location(label):
    if label in ['E1','M1F']:
        return '112'
    elif label in ['M1R','T1F']:
        return '011'
    elif label in ['MB1R','B1R','TM1F']:
        return '111'


if __name__ == '__main__':
    #filepath where results should be saved
    results_file = 'results.json'

    #hill coefficient
    n = 90 
    #decay rates
    decays = {'d_p' : 1, 'd_mc' : 1, 'd_mn' : 1} 
    #integration range
    t0 = 0
    tf = 100
    #number of initial conditions to sample
    num_IC = 10
    #number of threshold to sample
    num_theta_samples = 100
    
    param_file = 'Parameter_datasets/'
    param_labels = ['E1','M1F','M1R','MB1R','B1R','TM1F','T1F']
    results = dict([[label,{'num_large':0, 'num_small':0, 'large_params':[], 'small_params':[]}] for label in param_labels])
    results['anomalies'] = dict()

    for label in param_labels:
        print('Starting ' + label + ' classification.', flush = True)
        tic = time.time()
        params = make_parameters(label,num_theta_samples)
        # params = pd.read_csv(param_file + label + '_parameters.csv', header = None)
        # params = convert_to_dict_2(params) #params is a list of dictionaries
        IC_domain = IC_location(label) #state to sample initial conditions from 
        for k,p in enumerate(params):
            ICs = ic_function(IC_domain, p, num_IC) #randomly sampled initial conditions
            orbit_size = []
            for i in range(num_IC):
                try:
                    cur_IC = ICs[i,:]
                    sol = scipy.integrate.solve_ivp(lambda t,y:fun(t,y,p,n,decays),[t0,tf],cur_IC,method = 'BDF')
                    theta = theta_from_param(p)
                    dom_seq = get_periodic_domains(sol.y,theta,num_periods_to_verify = 2)
                    # if len(dom_seq)<4:
                    #     raise ValueError('No periodic orbit found at parameter ' + str(k) +' with IC ' + str(i))
                    orbit_size.append(classify_orbit(dom_seq))
                except:
                    print('\n\n\nparam index : ' + str(k) + ', IC index : ' + str(i) + '\n\n\n',flush = True)
                    raise ValueError
            orbit_size = np.unique(orbit_size)
            if len(orbit_size) == 1:
                size = orbit_size[0]
                if size == 'Small':
                    results[label]['num_small'] += 1
                    results[label]['small_params'].append(k)
                else: #size == 'Large'
                    results[label]['num_large'] += 1
                    results[label]['large_params'].append(k)
            else:
                if label in results['anomalies']:
                    results['anomalies'][label].append(k)
                else:
                    results['anomalies'][label] = [k]
        toc = time.time()
        timelapse = toc - tic
        print(label + ' classification done in ' + str(timelapse) + ' seconds.',flush = True)
        print('Number Large : ' + str(results[label]['num_large']) + ', Number Small : ' + str(results[label]['num_small']),flush = True)
    #save results. 
    with open(results_file,'w') as writer:
        json.dump(results,writer)