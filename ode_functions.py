"""
Some utility and plotting functions.
"""
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import pandas as pd
import scipy.integrate
from periodic_orbits import get_periodic_domains

#encodes the ordering of the variables for the plotting functions
VARIABLE_NAMES = ['Mc','Mn','P']


#######################
# Utilities for checking hysteresis in orbit lengths
#######################

def generate_all_lines(small_orbs, large_orbs, number):
    '''
    Takes 2 lists of parameters, generates number of linear parameters between all possible combinations of each.
    Returns: List of list of dictionaries.
    Args:
    small_orbs : (list) list of dictionaries of parameters which all result in small orbits
    large_orbs : (list) list of dictionaries of parameters which all result in large orbits (small, large
    interchangeable)
    number : number of linear parameters to generate between each pair
    '''
    line_list = []
    for i in small_orbs:
        for j in large_orbs:
            x = param_line(i,j,number)
            line_list.append(x)#should have length i*j
    return line_list

def check_line_list(line_list, n = 90, tf = 30, decays = {'d_p' : 1, 'd_mc' : 1, 'd_mn' : 1}):
    '''
    Calculates orbit lengths for each parameter set in a linear parameter object (element of line_list) by iterating
    through from one length orbit to other length. Saves lengths and order in list format. Calculates orbit lengths in
    reverse direction and saves result. Compares two lists to check for birythmicity. If same, result entry for
    that linear parameter object is False, if different, result is True. Returns resulting list.
    Args:
    line_list : (list) output of generate_all_lines(). List of list of dictionaries, each entry is list of parameter
    dictionaries connecting two different length orbits.
    n : Hill coefficient
    tf : ode solver time
    decays : (dictionary) decay rates for P, Mc, Mn
    '''
    interesting_lines = []
    for line in line_list:
        sizes = []
        #for the sake of easy ICs for first run
        param = line[0]
        sol = scipy.integrate.solve_ivp(fun, [0, tf], y0 = [param['thetaMnMc'], param['thetaPMn'], param['thetaMcP']], args = [param, n, decays], method = 'BDF')
        #create sizes, list of lenghts in forward direction
        for param in line:
            theta = theta_from_param(param)
            IC = sol.y[:,-1]
            sol = scipy.integrate.solve_ivp(fun, [0, tf], y0 = IC, args = [param, n, decays], method = 'BDF')
            output = get_periodic_domains(sol.y,theta,num_periods_to_verify = 2)
            size = classify_orbit(output)
            sizes.append(size)
            
    #create rev_rev_sizes, list of lengths in reverse direction
        # rev_line = line.copy()
        # rev_line.reverse()
        rev_sizes = []
        #for the sake of easy ICs for first run
        param = line[-1]
        sol = scipy.integrate.solve_ivp(fun, [0, tf], y0 = [param['thetaMnMc'], param['thetaPMn'], param['thetaMcP']], args = [param, n, decays], method = 'BDF')
        for param in reversed(line):
            theta = theta_from_param(param)
            rev_IC = sol.y[:,-1]
            sol = scipy.integrate.solve_ivp(fun, [0, tf], y0 = rev_IC, args = [param, n, decays], method = 'BDF')
            output = get_periodic_domains(sol.y,theta,num_periods_to_verify = 2)
            size = classify_orbit(output)
            rev_sizes.append(size)
        rev_sizes.reverse()
        
        #compare sizes to rev_rev_sizes. Interesting if different in two locations
        sizes = np.array(sizes)
        rev_sizes = np.array(rev_sizes)
        print(sizes,rev_sizes,flush = True)
        if sum(sizes != rev_sizes) > 1:
            interesting_lines.append(True)
        else:
            interesting_lines.append(False)
        # if sizes == rev_rev_sizes:
        #     result = False
        # elif sizes != rev_rev_sizes:
        #     result = True
        
    return interesting_lines

def generate_all_lines(small_orbs, large_orbs, number):
    '''
    Takes 2 lists of parameters, generates number of linear parameters between all possible combinations of each.
    Returns: List of list of dictionaries.
    Args:
    small_orbs : (list) list of dictionaries of parameters which all result in small orbits
    large_orbs : (list) list of dictionaries of parameters which all result in large orbits (small, large
    interchangeable)
    number : number of linear parameters to generate between each pair
    '''
    line_list = []
    for i in small_orbs:
        for j in large_orbs:
            x = param_line(i,j,number)
            line_list.append(x)#should have length i*j
    return line_list

def param_line(param1, param2, number):
    '''
    Returns length number list of dictionaries, linearly spaced between param1 and param2. Meant to be used to
    connect length 6 orbit parameters to length 8 orbit parameters to explore what happens in between.
    Args:
    param1 : one endpoint parameter (dictionary) (form of output of convert_to_dict)
    param2 : other endpoint parameter (dictionary)
    number : number of evenly spaced parameters between param1 and param2 desired
    '''
    import numpy as np
    t = np.linspace(0, 1, num = number, endpoint = True)#parametrization variable
    new_dictionaries = []#storage for 'number' of new dictionaries
    for i in t:#iterate through all spacings
        new_dict = {}
        for k in param1.keys():#alter each dictionary entry by same percentage
            new_dict[k] = i*param1[k] + (1-i)*param2[k]
        new_dictionaries.append(new_dict)
    return new_dictionaries
################
# Plotting utilities
##############

def theta_from_param(param):
    """
    Get the list of thresholds from param in the format required for get_periodic_domains. 

    :param param: parameter dictionary. 
    """
    if 'thetaMnMc' in param:
        return [[param['thetaMcMc'],param['thetaMnMc']], [param['thetaPMn']], [param['thetaPT'], param['thetaMcP']]]
    else: 
        return [[param['thetaMcMc']], [param['thetaPMn']], [param['thetaPT'], param['thetaMcP']]]


def plot_tseries_one_panel(sol,param,ax = None,plot_options = dict(),figsize = (10,5)):
    """
    Plot the ODE solution as a time series in a single panel. 

    :param sol: output of solve_ivp
    :param param: parameter dictionary 
    :param ax: (optional) matplotlib axes object on which to plot
    :param plot_options: (optional) dictionary of keyword arguments to pass to ax.plot
    :return: if ax is None returns fig, ax where fig is a matplotlib figure object and ax 
    is a matplotlib axes object. Otherwise returns None
    """
    var_names = VARIABLE_NAMES
    colors = ['b','g','r']
    if ax is None:
        ax_passed = False
        #create a figure and its ax
        fig,ax = plt.subplots(1,1,figsize = figsize)
    theta = theta_from_param(param)
    for i in range(3):
        #plot the time series
        ax.plot(sol.t,sol.y[i],color = colors[i])
        #plot the thresholds as horizontal lines
        xlim = ax.get_xlim()
        for threshold in theta[i]:
            ax.plot(xlim,[threshold]*2,color = colors[i],linestyle='--')
    if not ax_passed:
        return fig, ax

def plot_time_series_one_var(sol,param,y_index):
    fig, ax = plt.subplots(1,1)
    theta = theta_from_param(param)
    #plot the time series
    ax.plot(sol.t,sol.y[y_index])
    #plot the thresholds as horizontal lines
    xlim = ax.get_xlim()
    for threshold in theta[y_index]:
        ax.plot(xlim,[threshold]*2,'k')
    return fig,ax

def plot_time_series(sol,param,axs = None,plot_options = dict(),figsize = (16,4),make_ticks = True,label = True):
    """
    Plot the ODE solution as a time series. 

    :param sol: output of solve_ivp
    :param param: parameter dictionary
    :param axs: (optional) length 3 tuple of matplotlib axes objects on which to plot
    :param plot_options: (optional) dictionary of keyword arguments to pass to ax.plot
    :return: if axs is None returns fig, axs where fig is a matplotlib figure object and axs is a tuple
    of three matplotlib axes objects. Otherwise returns None
    """
    var_names = VARIABLE_NAMES
    if axs is None:
        axs_passed = False
        #create a figure and its axs
        fig, axs = plt.subplots(1,3,figsize = figsize)
    else:
        axs_passed = True
    theta = theta_from_param(param)
    for i in range(3):
        cur_ax = axs[i]
        #plot the time series
        cur_ax.plot(sol.t,sol.y[i])
        #plot the thresholds as horizontal lines
        xlim = cur_ax.get_xlim()
        for threshold in theta[i]:
            cur_ax.plot(xlim,[threshold]*2,'k')
        if label:
            cur_ax.set_ylabel(var_names[i])
            cur_ax.set_xlabel('t')
        if not make_ticks:
            cur_ax.set_xticks([])
            cur_ax.set_yticks([])
    if not axs_passed:
        return fig, axs


def plot_projections(sol,param,axs = None,plot_options = dict(),figsize = (12,5),make_ticks = True,label = True):
    """
    Plot two projections of the trajectory in phase space. The first is the (x,y)
    projection and the second is the (x,z) projection

    :param sol: output of solve_ivp
    :param param: parameter dictionary used by the ODE function 'fun'
    :param axs: (optional) tuple of matplotlib axes objects on which to plot. Creates a 
    new figure if this is not passed. 
    :param plot_options: (optional) key word arguments to pass to ax.plot
    :param figsize: (optional) tuple describing the figure size to pass to plt.subplots
    :return: if axs is not passed, returns fig, (ax1,ax2) where fig is a matplotlib figure object and ax1 and ax2
    are matplotlib axes objects. Otherwise, returns None
    """
    if axs is None:
        #create a figure and its axs
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = figsize)
    else:
        ax1, ax2 = axs
    plot_phase_projection(ax1,sol,param,x_index = 0,y_index = 1,plot_options = plot_options,make_ticks = make_ticks,label = label)
    plot_phase_projection(ax2,sol,param,x_index = 0,y_index = 2,plot_options = plot_options,make_ticks = make_ticks,label = label)
    if axs is None:
        return fig, (ax1,ax2)


def plot_phase_projection(ax,sol,param,x_index,y_index,plot_options = dict(),make_ticks = True,label = True):
    """
    Plot a projection of the trajectory in phase space.

    :param ax: a matplotlib axes object
    :param sol: output of solve_ivp 
    :param param: parameter dictionary used by the ODE function 'fun'
    :param x_index: index of variable to plot on the x-axis
    :param y_index: index of variable to plot on the y-axis
    :param plot_options: (optional) key word arguments to pass to ax.plot
    """
    var_names = VARIABLE_NAMES
    #plot trajectory
    ax.plot(sol.y[x_index],sol.y[y_index],**plot_options)
    #plot thresholds. This should probably label the threshold lines, but I'm lazy
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    theta = theta_from_param(param)
    #x thresholds
    for threshold in theta[x_index]:
        ax.plot([threshold]*2,ylim,'k')
    if not make_ticks:
        ax.set_xticks([])
    #y thresholds
    for threshold in theta[y_index]:
        ax.plot(xlim,[threshold]*2,'k')
    if not make_ticks:
        ax.set_yticks([])
    #set x and y labels
    if label:
        ax.set_xlabel(var_names[x_index])
        ax.set_ylabel(var_names[y_index])

######################
# ODE simulation utilities
#######################

def ic_function(domain, params,n):
    """
    Returns n randomly generated initial condition in desired region. Seed to 
    random number generator is reset each use. 
    
    Args:
    domain: (string) indicating region in parameter space from which to select initial condition (ex "000")
    params: (dictionary) of parameter values
    n: (integer) number of initial conditions to generate
    """
    #set seed for reproducibility
    np.random.seed(0)
    #extract thresholds
    (thetaMcP, thetaPT, thetaMcMc, thetaPMn) = (params['thetaMcP'], params['thetaPT'], params['thetaMcMc'], 
                params['thetaPMn'])
    
    if 'thetaMnMc' in params:
        #threshold is unfolded
        thetaMnMc = params['thetaMnMc']
    else:
        #threshold is not unfolded
        thetaMnMc = thetaMcMc
    
    #get sorted unique values of thresholds for each variable. 0 is lower bound
    thresh_Mc = list(np.unique([0,thetaMnMc,thetaMcMc]))
    thresh_Mn = list(np.unique([0,thetaPMn]))
    thresh_P = list(np.unique([0,thetaMcP,thetaPT]))
    
    #add upper bound in each variable
    for thresh in [thresh_Mc,thresh_Mn,thresh_P]:
        thresh.append(2*thresh[-1])
    all_thresh = [thresh_Mc,thresh_Mn,thresh_P]
    
    #initialize array of initial conditions. Each row is an initial condition
    ICs = np.zeros((n,3))
    #randomly select initial conditions from domain
    for i,value in enumerate(domain):
        state = int(value)
        cur_thres = all_thresh[i]
        ICs[:,i] = np.random.uniform(low = cur_thres[state],high = cur_thres[state+1],size = (n))
    return ICs


def fun(t, y, params, n, decays):
    """
    DSGRN p53 Mdm2 system.
    
    Args:
    t: scalar
    y: vars
    params: (dictionary) of parameter values
    decays: (dictionary) of decay parameters, d_p, d_mc, d_mn
    """
    #extract parameters
    (thetaMcP, thetaPT, thetaMcMc, thetaPMn, LMcP, UMcP, 
        LPT,UPT, LMcT, UMcT, LPMn, UPMn) = (params['thetaMcP'], params['thetaPT'], 
            params['thetaMcMc'], params['thetaPMn'], params['LMcP'], params['UMcP'], 
            params['LPT'], params['UPT'], params['LMcT'], params['UMcT'], 
            params['LPMn'], params['UPMn'])

    if 'thetaMnMc' in params:
        #threshold is unfolded
        thetaMnMc = params['thetaMnMc']
    else:
        #threshold is not unfolded
        thetaMnMc = thetaMcMc
    
    (d_p, d_mc, d_mn) = (decays['d_p'], decays['d_mc'], decays['d_mn'])
    
    #extract variables
    mc, mn, p = y
    #right hand side
    f = [(LMcP + ((UMcP - LMcP)*p**n)/(thetaMcP**n + p**n)) - 
              (LPT + ((UPT - LPT)*thetaPT**n)/(thetaPT**n + p**n))*
              (LMcT + ((UMcT - LMcT)*mc**n)/(thetaMcMc**n + mc**n)) - 
              d_mc*mc,
         (LPT + ((UPT - LPT)*thetaPT**n)/(thetaPT**n + p**n))*
              (LMcT + ((UMcT - LMcT)*mc**n)/(thetaMnMc**n + mc**n)) - d_mn*mn,
    (LPMn + ((UPMn - LPMn)*thetaPMn**n)/(thetaPMn**n + mn**n) - d_p)*p]

    return f

#########################
# Trajectory analysis utilities
##########################

def classify_orbit(domains):
    """
    Returns label of "Large" or "Small" based on the thresholds traversed by P. 
    Args:
    domains: output of get_periodic_domains
    """
    seen_low = False
    seen_high = False
    for state in domains:
        if state[2] == 2:
            seen_high = True
        elif state[2] == 0:
            seen_low = True
        if seen_high and seen_low:
            return "Large"
    return "Small"

def lower_double_id(output):
    '''
    Returns label of "Large" or "Small" for a particular trajectory for parameter node where we expect birythmicity
    to be of lower inner loop type.
    Args:
    output : list of arrays identifying domains through which trajectory passes (result of 
    get_periodic_domains(trajectory))
    '''
    newoutput = []#convert output from arrays to strings for easier checks
    for i in output:
        string = ''
        for j in i:
            string = string + str(j)
        newoutput.append(string)
        
    top_domains = ['202','102','002','212','112','012']
    for domain in top_domains:
        if domain in newoutput:
            return 'Large'
        else:
            pass
    return 'Small'
        
def upper_double_id(output):
    '''
    Returns label of "Large" or "Small" for a particular trajectory for parameter node where we expect birythmicity
    to be of upper inner loop type.
    Args:
    output : list of arrays identifying domains through which trajectory passes (result of 
    get_periodic_domains(trajectory))
    '''
    newoutput = []#convert output from arrays to strings for easier checks
    for i in output:
        string = ''
        for j in i:
            string = string + str(j)
        newoutput.append(string)
        
    top_domains = ['200','100','000','210','110','010']
    for domain in top_domains:
        if domain in newoutput:
            return 'Large'
        else:
            pass
    return 'Small'

def removeElements(A, B): 
    n = len(A) 
    return any(A == B[i:i + n] for i in range(len(B)-n + 1))


def pathfinder(trajectory, params):
    """
    Returns path through state transition graph.
    Args:
    trajectory: solution output of an ode solver. (sol.y)
    params: (dictionary) of parameter values, must include LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc,
            thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc, UMcP, UMcPT, UMnMc, UMnP, UPMn
    """
    
    from itertools import groupby
    
    (LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc, thetaMnMc, thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc, 
     UMcP, UMcPT, UMnMc, UMnP, UPMn) = (params['LMcMc'], params['LMcP'], params['LMcPT'], params['LMnMc'],
                                  params['LMnP'], params['LPMn'], params['thetaMcMc'], params['thetaMnMc'], 
                                  params['thetaMcP'], params['thetaMcPT'], params['thetaMnMc'],
                                  params['thetaPMn'], params['UMcMc'], params['UMcP'], params['UMcPT'], 
                                  params['UMnMc'], params['UMnP'], params['UPMn'])
    
    thetaMnP = thetaMcPT #always true
    
    mc_vec = trajectory[0]
    mn_vec = trajectory[1]
    p_vec = trajectory[2]
    
    
#     p_vec = trajectory[0] #original order
#     mc_vec = trajectory[1]
#     mn_vec = trajectory[2]
    
    #list of exact points in phase space for each time step

    points = []
    for i in range(len(p_vec)):
        points.append([p_vec[i], mc_vec[i], mn_vec[i]])
    
    #region is list of region each point falls into
    
    regions = []
    for point in points:
        if point[0] < thetaMcP:#bottom layer
            if point[1] < thetaMcMc:#back
                if point[2] < thetaPMn:#left
                    regions.append("000")
                elif point[2] > thetaPMn:#right
                    regions.append("010")
            elif point[1] > thetaMcMc and point[1] < thetaMnMc:#middle
                if point[2] < thetaPMn:#left
                    regions.append("mid BL")
                elif point[2] > thetaPMn:#right
                    regions.append("mid BR")
            elif point[1] > thetaMnMc:#front
                if point[2] < thetaPMn:#left
                    regions.append("100")
                elif point[2] > thetaPMn:#right
                    regions.append("110")
        elif point[0] > thetaMcP and point[0] < thetaMnP:#middle layer
            if point[1] < thetaMcMc:#back
                if point[2] < thetaPMn:#left
                    regions.append("001")
                elif point[2] > thetaPMn:#right
                    regions.append("011")
            elif point[1] > thetaMcMc and point[1] < thetaMnMc:#middle
                if point[2] < thetaPMn:#left
                    regions.append("mid ML")
                elif point[2] > thetaPMn:#right
                    regions.append("mid MR")
            elif point[1] > thetaMnMc:#front
                if point[2] < thetaPMn:#left
                    regions.append("101")
                elif point[2] > thetaPMn:#right
                    regions.append("111")
        elif point[0] > thetaMnP:#top
            if point[1] < thetaMcMc:#back
                if point[2] < thetaPMn:#left
                    regions.append("002")
                elif point[2] > thetaPMn:#right
                    regions.append("012")
            elif point[1] > thetaMcMc and point[1] < thetaMnMc:#middle
                if point[2] < thetaPMn:#left
                    regions.append("mid TL")
                elif point[2] > thetaPMn:#right
                    regions.append("mid TR")
            elif point[1] > thetaMnMc:#front
                if point[2] < thetaPMn:#left
                    regions.append("102")
                elif point[2] > thetaPMn:#right
                    regions.append("112")

    #cleanregions is simplified list - eliminates consecutive repeats to illustrate path
    cleanregions = [i[0] for i in groupby(regions)]
    
    return cleanregions

###################################
# Parameter utilities
#################################

def Mc_in(p,McP,PT,McT):
    """
    Get the value of the input to Mc
    Input:
        p - parameter dictionary
        McP,PT,McT - (string) one of 'L' or 'U'
    """
    return p[McP + 'McP'] - p[PT + 'PT'] * p[McT + 'McT']

def make_parameters(label,num_theta_samples):
    """
    Create a list of parameters. Each parameter in the associated parameter file is used to specify
    L and U values and theta values are sampled num_theta_samples based on their range. 

    Input:
        label: (string) label of the parameter node
        num_theta_samples: (int) number of times to sample theta from each parameter in param_file
    """
    np.random.seed(0) #set seed for reproducibility
    param_directory = 'Parameter_datasets/'
    LU_params = pd.read_csv(param_directory + label + '_parameters.csv', header = None)
    LU_params = convert_to_dict_2(LU_params) #list of parameter dictionaries. 
    param_list = []
    for p in [LU_params[0]]:
        #P and Mn threshold bounds are independent of parameter node
        thetaMcP_samples = np.random.uniform(low = p['LPMn'],high = p['UPMn'],size = (num_theta_samples))
        thetaPT_samples = np.random.uniform(low = thetaMcP_samples,high = p['UPMn'],size = (num_theta_samples))
        thetaPMn_samples = np.random.uniform(low = 100/95*p['UPT']*p['LMcT'],high = 100/105*p['LPT']*p['UMcT'],size = (num_theta_samples))
        #Mc threshold bounds depend on the parameter node
        if label == 'T1F':
            #values below the lower threshold
            lower_vals = [Mc_in(p,*val) for val in ['LUU','LUL','UUU','ULU','UUL']]
            #values above the larger threshold
            upper_vals = [Mc_in(p,*val) for val in ['ULL']]
            T_low = 'Mn' #lower target node of Mc
            T_high = 'Mc' #higher target node of Mc
        elif label == 'MB1R':
            lower_vals = [Mc_in(p,*val) for val in ['LUU','UUU']]
            upper_vals = [Mc_in(p,*val) for val in ['LUL','UUL','ULU','ULL']]
            T_low = 'Mc'
            T_high = 'Mn'
        elif label == 'TM1F':
            lower_vals = [Mc_in(p,*val) for val in ['LUU','UUU','LUL','ULU']]
            upper_vals = [Mc_in(p,*val) for val in ['UUL','ULL']]
            T_low = 'Mn'
            T_high = 'Mc'
        elif label == 'B1R':
            lower_vals = [Mc_in(p,*val) for val in ['LUU']]
            upper_vals = [Mc_in(p,*val) for val in ['UUU','LUL','UUL','ULU','ULL']]
            T_low = 'Mc'
            T_high = 'Mn'
        elif label == 'M1R':
            lower_vals = [Mc_in(p,*val) for val in ['LUU','LUL','UUU']]
            upper_vals = [Mc_in(p,*val) for val in ['UUL','ULU','ULL']]
            T_low = 'Mc'
            T_high = 'Mn'
        elif label == 'M1F':
            lower_vals = [Mc_in(p,*val) for val in ['LUU','LUL','UUU']]
            upper_vals = [Mc_in(p,*val) for val in ['UUL','ULU','ULL']]
            T_low = 'Mn'
            T_high = 'Mc'
        elif label == 'E1':
            lower_vals = [Mc_in(p,*val) for val in ['LUU','LUL']]
            upper_vals = [Mc_in(p,*val) for val in ['UUU','ULU','UUL','ULL']]
        if label != 'E1':
            theta_Mc_lower = 1.05*max(lower_vals) #lower bound of lower threshold
            theta_Mc_low_upper = 95*100/(105**2)*min(upper_vals) #upper bound for smaller threshold
            theta_Mc_high_upper = 100/105*min(upper_vals) #upper bound for larger threshold
            spread_factor = 105/95 #minimum ratio of larger threshold to lower threshold
            #sample Mc thresholds
            theta_low_Mc_samples = np.random.uniform(low = theta_Mc_lower,high = theta_Mc_low_upper,size = (num_theta_samples))
            theta_high_Mc_samples = np.random.uniform(low = spread_factor*theta_low_Mc_samples,high = theta_Mc_high_upper,size = (num_theta_samples))
        else: # E1 has thetaMcMc = thetaMnMc
            theta_Mc_lower = 1.05*max(lower_vals)
            theta_Mc_upper = 100/105*min(upper_vals)
            theta_low_Mc_samples = np.random.uniform(low = theta_Mc_lower,high = theta_Mc_upper,size = (num_theta_samples))
            theta_high_Mc_samples = theta_low_Mc_samples #thresholds are the same. 
            #arbitrary assignment of low and high targets
            T_low = 'Mn'
            T_high = 'Mc'
        for i in range(num_theta_samples):
            q = copy.deepcopy(p)
            q['thetaMcP'] = thetaMcP_samples[i]
            q['thetaPT'] = thetaPT_samples[i]
            q['thetaPMn'] = thetaPMn_samples[i]
            q['theta' + T_low + 'Mc'] = theta_low_Mc_samples[i]
            q['theta' + T_high + 'Mc'] = theta_high_Mc_samples[i]
            param_list.append(q)
    return param_list
         


    


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        frac = float(num) / float(denom)
        return frac

def convert_to_dict(data):
    """
    Returns dictionary of parameters as keys, values. (Use as params throughout file)
    Args:
    data: Pandas dataframe with single column string entry of form "LMcP -> 3"
    """
    newdata = []
    for i in range(len(data)):
        newdata.append(data["0"][i].split(" -> "))
    for i in range(len(newdata)):
        newdata[i][1] = convert_to_float(newdata[i][1])
    params = dict(newdata)
    return params

def convert_to_dict_2(data):
    """
    Returns list of dictionaries of parameters as keys, values. (Use as params throughout file)
    Args:
    data: Pandas dataframe with parameter sets as rows. Entries of form "LMcP -> 3"
    """
    alldata = []
    for i in range(len(data)):
        paramset = data.iloc[i,:].to_frame(name = "0")
        paramsetdict = convert_to_dict(paramset)
        alldata.append(paramsetdict)
    return alldata

##############################

def get_parameter_node(parameters):
    """
    Returns parameter node into which a given set of parameters falls.
    Args:
        parameters: (dictionary) output obtained from convert_to_dict for a given parameter set.
    """
    return_string = ''
    if parameters['thetaMcP'] < parameters['thetaMcPT']:#P1 or P2
        return_string = return_string + 'P1'
        if parameters['thetaMcMc'] == parameters['thetaMnMc']:#test all (except two extremes) of the _-_ _ type words against threshold
            return_string = return_string + 'No_BW'#no splitting of threshold - only when no BW
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'B_BW'
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'MB_BW'
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'TMB_BW'
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'M_BW'
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'TM_BW'
            return return_string
        elif parameters['UMcP']-parameters['UMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'T_BW'
            return return_string
        else:
            print('Unidentified parameter node.')
        
    elif parameters['thetaMcPT'] < parameters['thetaMcP']:#other P ordering
        return_string = return_string + 'P2'
        if parameters['thetaMcMc'] == parameters['thetaMnMc']:#test all (except two extremes) of the _-_ _ type words against threshold
            return_string = return_string + 'No_BW'#no splitting of threshold - only when no BW
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc']:
            return_string = return_string + 'B_BW'
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'MB_BW'
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'TMB_BW'
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] > parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'M_BW'
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] > parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'TM_BW'
            return return_string
        elif parameters['LMcP']-parameters['UMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['LMcMc'] < parameters['thetaMnMc'] and parameters['UMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc'] and parameters['LMcP']-parameters['LMcPT']*parameters['UMcMc'] < parameters['thetaMnMc']:
            return_string = return_string + 'T_BW'
            return return_string
        else:
            print('Unidentified parameter node.')

def perturb_parameter_node(parameters):
    """
    Returns a new parameter node which is a 'small' perturbation of parameters.
    Args:
        parameters: (dictionary) output obtained from convert_to_dict for a given parameter set.
    """
    new_params = {}
    for key in params.keys():
        perturbation = random.random()#will we want to change by more than 100%/is this way too much?
        sign = random.choice([-1,1])
        newval = sign*perturbation*params[key]
        new_params[key] = newval
        #check positivity
    return new_params

def get_ic_domain(ic, params):
    """
    Returns STG node in which ic initial condition falls for a given set of parameters.
    
    Args:
    ic: (list) indicating initial condition to be located, order [mc, mn, p] (output of ic_function)
    params: (dictionary) of parameter values, must include LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc,
            thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc, UMcP, UMcPT, UMnMc, UMnP, UPMn
    """
    import random
    
    
    (LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc, thetaMnMc, thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc,
     UMcP, UMcPT, UMnMc, UMnP, UPMn) = (params['LMcMc'], params['LMcP'], params['LMcPT'], params['LMnMc'],
                                  params['LMnP'], params['LPMn'], params['thetaMcMc'], params['thetaMnMc'], 
                                  params['thetaMcP'], params['thetaMcPT'], params['thetaMnMc'],
                                  params['thetaPMn'], params['UMcMc'], params['UMcP'], params['UMcPT'], 
                                  params['UMnMc'], params['UMnP'], params['UPMn'])
    
    thetaMnP = thetaMcPT #always true
    
    if ic[2] < thetaMcP:#bottom layer
        if ic[0] < thetaMcMc:#back
            if ic[1] < thetaPMn:#left
                region = "000"
            elif ic[1] > thetaPMn:#right
                region = "010"
        elif ic[0] > thetaMcMc and ic[0] < thetaMnMc:#middle
            if ic[1] < thetaPMn:#left
                region = "mid BL"
            elif ic[1] > thetaPMn:#right
                region = "mid BR"
        elif ic[0] > thetaMnMc:#front
            if ic[1] < thetaPMn:#left
                region = "100"
            elif ic[1] > thetaPMn:#right
                region = "110"
    elif ic[2] > thetaMcP and ic[2] < thetaMnP:#middle layer
        if ic[0] < thetaMcMc:#back
            if ic[1] < thetaPMn:#left
                region = "001"
            elif ic[1] > thetaPMn:#right
                region = "011"
        elif ic[0] > thetaMcMc and ic[0] < thetaMnMc:#middle
            if ic[1] < thetaPMn:#left
                region = "mid ML"
            elif ic[1] > thetaPMn:#right
                region = "mid MR"
        elif ic[0] > thetaMnMc:#front
            if ic[1] < thetaPMn:#left
                region = "101"
            elif ic[1] > thetaPMn:#right
                region = "111"
    elif ic[2] > thetaMnP:#top
        if ic[0] < thetaMcMc:#back
            if ic[1] < thetaPMn:#left
                region = "002"
            elif ic[1] > thetaPMn:#right
                region = "012"
        elif ic[0] > thetaMcMc and ic[0] < thetaMnMc:#middle
            if ic[1] < thetaPMn:#left
                region = "mid TL"
            elif ic[1] > thetaPMn:#right
                region = "mid TR"
        elif ic[0] > thetaMnMc:#front
            if ic[1] < thetaPMn:#left
                region = "102"
            elif ic[1] > thetaPMn:#right
                region = "112"
        
    return region
