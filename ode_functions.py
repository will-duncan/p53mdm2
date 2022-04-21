"""
Some utility and plotting functions.
"""
import matplotlib.pyplot as plt

#encodes the ordering of the variables for the plotting functions
VARIABLE_NAMES = ['Mc','Mn','P']

def theta_from_param(param):
    """
    Get the list of thresholds from param in the format required for get_periodic_domains. 

    :param param: parameter dictionary. 
    """
    return [[param['thetaMcMc'],param['thetaMnMc']], [param['thetaPMn']], [param['thetaMcPT'], param['thetaMcP']]]


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



"""
Functions written by Erika for use with Example_odd_orbits.ipynb
"""
def ic_function(domain, params):
    """
    Returns a randomly generated initial condition in desired region.
    
    Args:
    domain: (string) indicating region in parameter space from which to select initial condition (ex "000")
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
    
    if domain == "000":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(0, min_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "010":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(0, min_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "110":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(0, min_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "100":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(0, min_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "001":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(min_p, max_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "011":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(min_p, max_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "111":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(min_p, max_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "101":
        min_p = min(thetaMcP, thetaMcPT, thetaMnP)
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(min_p, max_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "002":
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(max_p, 2*max_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "012":
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        min_mc = min(thetaMcMc, thetaMnMc)
        p = random.uniform(max_p, 2*max_p)
        mc = random.uniform(0, min_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "112":
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(max_p, 2*max_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(thetaPMn, 2*thetaPMn)
        w0 = [mc, mn, p]
    elif domain == "102":
        max_p = max(thetaMcP, thetaMcPT, thetaMnP)
        max_mc = max(thetaMcMc, thetaMnMc)
        p = random.uniform(max_p, 2*max_p)
        mc = random.uniform(max_mc, 2*max_mc)
        mn = random.uniform(0, thetaPMn)
        w0 = [mc, mn, p]
        
    return w0

def fun(t, y, params, n, decays):
    
    """
    DSGRN p53 Mdm2 system.
    
    Args:
    t: scalar
    y: vars
    params: (dictionary) of parameter values, must include LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc,
            thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc, UMcP, UMcPT, UMnMc, UMnP, UPMn
    decays: (dictionary) of decay parameters, d_p, d_mc, d_mn
    """
    
    (LMcMc, LMcP, LMcPT, LMnMc, LMnP, LPMn, thetaMcMc, thetaMnMc, thetaMcP, thetaMcPT, thetaMnMc, thetaPMn, UMcMc, 
     UMcP, UMcPT, UMnMc, UMnP, UPMn) = (params['LMcMc'], params['LMcP'], params['LMcPT'], params['LMnMc'],
                                  params['LMnP'], params['LPMn'], params['thetaMcMc'], params['thetaMnMc'], 
                                  params['thetaMcP'], params['thetaMcPT'], params['thetaMnMc'],
                                  params['thetaPMn'], params['UMcMc'], params['UMcP'], params['UMcPT'], 
                                  params['UMnMc'], params['UMnP'], params['UPMn'])
    
    thetaMnP = thetaMcPT #always true
    
    (d_p, d_mc, d_mn) = (decays['d_p'], decays['d_mc'], decays['d_mn'])
    
    mc, mn, p = y
    
    f = [(LMcP + ((UMcP - LMcP)*p**n)/(thetaMcP**n + p**n)) - 
              (LMcPT + ((UMcPT - LMcPT)*thetaMcPT**n)/(thetaMcPT**n + p**n))*
              (LMcMc + ((UMcMc - LMcMc)*mc**n)/(thetaMcMc**n + mc**n)) - 
              d_mc*mc,
         (LMnP + ((UMnP - LMnP)*thetaMnP**n)/(thetaMnP**n + p**n))*
              (LMnMc + ((UMnMc - LMnMc)*mc**n)/(thetaMnMc**n + mc**n)) - d_mn*mn,
    (LPMn + ((UPMn - LPMn)*thetaPMn**n)/(thetaPMn**n + mn**n) - d_p)*p]

#     p, mc, mn = y #original order
    
#     f = [(LPMn + ((UPMn - LPMn)*thetaPMn**n)/(thetaPMn**n + mn**n) - d_p)*p,
#              (LMcP + ((UMcP - LMcP)*p**n)/(thetaMcP**n + p**n)) - 
#                   (LMcPT + ((UMcPT - LMcPT)*thetaMcPT**n)/(thetaMcPT**n + p**n))*
#                   (LMcMc + ((UMcMc - LMcMc)*mc**n)/(thetaMcMc**n + mc**n)) - 
#                   d_mc*mc,
#              (LMnP + ((UMnP - LMnP)*thetaMnP**n)/(thetaMnP**n + p**n))*
#                   (LMnMc + ((UMnMc - LMnMc)*mc**n)/(thetaMnMc**n + mc**n)) - d_mn*mn] #original order
    
    return f

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
