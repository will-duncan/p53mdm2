
import numpy as np
import warnings

#this needs to be tuned. Should be the same order of magnitude as the tolerance for the ode solver. 
DEFAULT_TOLERANCE = 5e-3  

###########################
# Detecting periodicity
###########################
def is_periodic(ode_func,trajectory,tol = DEFAULT_TOLERANCE):
    """
    Determine if a trajectory is periodic. Assumes that the ODE which generated the
    ODE is dissapative and doesn't have a chaotic attractor. 

    :param ode_func: handle to the ODE function which generated the trajectory
    :param trajectory: solution output of an ode solver. Asumes trajectory is an 
    ndarray with shape (n,n_time_points)
    :param tol: (optional) tolerance to pass to is_equilibrium.
    """
    if is_equilibrium(ode_func,trajectory[:,-1],tol = tol):
        return False
    return True

def is_equilibrium(ode_func,x,tol = DEFAULT_TOLERANCE):
    """
    Determine if x is an equilibrium of an autonomous ODE. 

    :param ode_func: function which gives the right hand side of an autonomous ODE. 
    Must have calling signature ode_func(t,x)
    :param x: list or numpy vector with real valued entries.  
    :param tol: (optional) error tolerance. 
    :return: Returns True if the 2-norm of ode_func is less than tol at x. 
    """
    if np.linalg.norm(ode_func(0,x)) < tol:
        return True
    return False

###############################################
# Getting domains traversed by a periodic orbit
###############################################

def get_periodic_domains(trajectory,theta,num_periods_to_verify = 2):
    """
    Get the list of domains the periodic orbit contained in a trajectory traversed. 

    :param trajectory: solution output of an ode solver. Asumes trajectory is an 
    ndarray with shape (n,n_time_points)
    :param theta: list of lists of theta values. theta[j] must be the values of 
    theta_{ij} for all nodes i in Targets(j). Requires len(theta) == trajectory.shape[0]
    :param num_periods_to_verify: (optional, default = 2) If a coordinate is repeated, 
    it does not guarantee that the sequence seen so far is the correct sequence of domains. 
    This parameter determines how many times a sequence needs to be repeated in order 
    to return the sequence. 
    """
    n_time_points = trajectory.shape[1]
    rev_trajectory = np.fliplr(trajectory)
    # get the last well defined coordinate
    last_coordinate = [-1]
    i = 0
    while -1 in last_coordinate:
        last_coordinate = stg_coordinate(rev_trajectory[:,i],theta)
        i += 1
    rev_sequence = [last_coordinate]
    periodic_candidate_sequence = [last_coordinate]
    num_periods_seen = 0
    for t in range(i,n_time_points):
        prev_coordinate = rev_sequence[-1]
        cur_coordinate = stg_coordinate(rev_trajectory[:,t],theta)
        defined_indices = cur_coordinate != -1
        if np.array_equal(cur_coordinate[defined_indices],prev_coordinate[defined_indices]):
            #coordinate is repeated, do nothing
            continue
        else:
            #new coordinate, add it to the sequence
            #replace indeterminate coordinate entries with the previous coordinate's
            cur_coordinate[~defined_indices] = prev_coordinate[~defined_indices]
            rev_sequence.append(cur_coordinate)
        if np.array_equal(cur_coordinate,rev_sequence[0]) and (len(rev_sequence)-1) % len(periodic_candidate_sequence) == 0:
            #current coordinate matches first coordinate and the lengths make sense 
            #for there to be multiple periods in rev_sequence
            cand_length = len(periodic_candidate_sequence)
            orbit_start_index = num_periods_seen*cand_length
            if cand_length == len(rev_sequence[orbit_start_index:-1]) and \
               all([np.array_equal(periodic_candidate_sequence[i],rev_sequence[orbit_start_index+i]) for i in range(cand_length)]):
            #periodic_candidate_sequence == rev_sequence[num_periods_seen*len(periodic_candidate_sequence):-1]:
                num_periods_seen += 1
                if num_periods_seen == num_periods_to_verify:
                    #sequence is periodic and verified. end loop
                    break
            else: #current candidate sequence is proved not to be the sequence
                #update candidate sequence
                periodic_candidate_sequence = rev_sequence[:-1]
                num_periods_seen = 1
    # if t == n_time_points-1:
    #     warnings.warn('get_periodic_domains looked at all time points but did not verify a periodic sequence to desired verification level of num_periods_to_verify = {}. The trajectory may not be periodic.'.format(num_periods_to_verify))
    periodic_candidate_sequence.reverse()
    return periodic_candidate_sequence


def stg_coordinate(x,theta,tol = DEFAULT_TOLERANCE):
    """
    Get the coordinates of x in the state transition graph whose cells are defined
    by theta. If x[j] is within the tolerance of a threshold, the jth coordinate is -1.

    :param x: list or numpy vector with real valued entries. Requires len(x) == len(theta)
    :param theta: list of lists of theta values. theta[j] must be the values of 
    theta_{ij} for all nodes i in Targets(j).
    :param tol: (optional) the coordinate is indeterminate if x is within tol of a threshold. 
    tol should be the same order of magnitude as the ODE solver tolerances.  
    :return: np.array with length equal to the length of x
    """
    coordinate = np.zeros(len(x),dtype = int)
    for j in range(len(x)):
        theta_j = np.array(sorted(theta[j]))
        x_larger_indices = np.nonzero(x[j] > theta_j)[0]
        if len(x_larger_indices) == 0: #x[j] is smaller than all theta_ij
            if x[j] > theta_j[0] - tol: #x[j] is within the tolerance band of the first threshold
                coordinate[j] = -1
            else: #x[j] outside of tolerance band
                coordinate[j] = 0
        else:
            cur_coord = x_larger_indices[-1] + 1
            if x[j] < theta_j[cur_coord - 1] + tol: #x[j] is within tolerance band of the threshold below it
                coordinate[j] = -1
            elif cur_coord < len(theta_j) and x[j] > theta_j[cur_coord] - tol: #x[j] is within tolerance band of the threshold above it
                coordinate[j] = -1
            else: #x[j] outside of all tolerance bands. 
                coordinate[j] = cur_coord
    return coordinate


#####################################
# tests
#####################################
from scipy.integrate import solve_ivp
import ode_functions as ode
import pandas as pd


def test():
    def not_periodic_ode(t,y):
        return -y
    def periodic_ode(t,y):
        return np.array([y[1],-y[0]])

    #is_equilibrium and is_periodic
    assert(is_equilibrium(not_periodic_ode,0))
    not_periodic_soln = solve_ivp(not_periodic_ode,[0,100],[1,4])
    assert(not is_periodic(not_periodic_ode,not_periodic_soln.y))
    periodic_soln = solve_ivp(periodic_ode,[0,100],[2,3])
    assert(is_periodic(periodic_ode,periodic_soln.y))

    #stg_coordinate
    theta = [[1,2,3],[0,1,2,5]]
    assert(np.array_equal(stg_coordinate([.5,.5],theta),[0,1]))
    assert(np.array_equal(stg_coordinate([2.3,6],theta), [2,4] ))
    assert(np.array_equal(stg_coordinate([1+1e-4,10],theta), [-1,4]))

    #get_periodic_domains
    soln = solve_ivp(repressilator,[0,100],[2,5,.5],t_eval = np.linspace(60,100,1000))
    assert(is_periodic(repressilator,soln.y))
    theta = [[1],[1],[1]]
    sequence = get_periodic_domains(soln.y,theta)
    expected = [(1,0,0),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0)]
    check_sequence(sequence,expected)
    theta = [[.9,1],[1],[1]]
    sequence = get_periodic_domains(soln.y,theta)
    expected = [(2,0,1),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(2,1,0),(2,0,0)]
    check_sequence(sequence,expected)

def test_erratic_examples():
    ICs = [[0.4892486171059155, 0.4575516379632611, 0.4971665381268848],
            [0.542199258010232, 0.828179826803997, 0.13748879802025882],
            [0.17954843891491512, 0.48566393161534516, 0.256121302597364],
            [0.26449995810414384, 0.2643148553782599, 0.1720579468713308],
            [0.8588205546662672, 0.7309747595503258, 0.5303639877911169],
            [0.7763016136243949, 0.15969742584153765, 0.22905766734687083],
            [0.4761628530431625, 0.2767974724547933, 0.30091218518935814],
            [0.8863598397610741, 0.7986174204320661, 0.37025990074120774],
            [0.3416236293766618, 0.1513810025773655, 0.2514467467152447],
            [0.1898517304057531, 0.06600814025927429, 0.3217165203352351]]
    tf = 35
    n = 90
    decays = {'d_p' : 1, 'd_mc' : 1, 'd_mn' : 1}
    # test an unerratic parameter with length 6 orbits
    len_6_data = pd.read_csv("Parameter_datasets/P1BBWnode1.csv", header = None)
    len_6_params = ode.convert_to_dict(len_6_data)
    for IC in ICs:
        sol = solve_ivp(lambda t,y: ode.fun(t,y,len_6_params,n,decays), [0,tf], y0 = IC, method = 'BDF')
        assert(len(get_periodic_domains(sol.y,ode.theta_from_param(len_6_params),num_periods_to_verify = 2)) == 6)
    # length 4 orbit is expected?
    len_4_data = pd.read_csv("Parameter_datasets/P1BBWnode3.csv", header = None)
    len_4_params = ode.convert_to_dict(len_4_data)
    for IC in ICs:
        sol = solve_ivp(lambda t,y: ode.fun(t,y,len_4_params,n,decays), [0,tf], y0 = IC, method = 'BDF')
        domains = get_periodic_domains(sol.y,ode.theta_from_param(len_4_params))
        check_sequence(domains,[(1,0,0),(1,0,1),(1,1,1),(1,1,0)])
    # previously erratic orbit lengths
    erratic_data_1 = pd.read_csv("Parameter_datasets/P1BBWnode7.csv", header = None)
    erratic_data_2 = pd.read_csv("Parameter_datasets/P1BBWnode8.csv", header = None)
    erratic_param_1 = ode.convert_to_dict(erratic_data_1)
    erratic_param_2 = ode.convert_to_dict(erratic_data_2)
    #erratic_param_1 should give a length 6 orbit based on a manual check in Example_odd_orbits.ipynb
    for IC in ICs:
        sol = solve_ivp(lambda t,y: ode.fun(t,y,erratic_param_1,n,decays), [0,tf], y0 = IC, method = 'BDF')
        domains = get_periodic_domains(sol.y,ode.theta_from_param(erratic_param_1),num_periods_to_verify = 2)
        check_sequence(domains,[(1,1,0),(1,0,0),(1,0,1),(1,0,2),(1,1,2),(1,1,1)])
    
    #erratic_param_2 should give length 4 orbits based on a manual check in Example_odd_orbits.ipynb
    for IC in ICs:
        sol = solve_ivp(lambda t,y: ode.fun(t,y,erratic_param_2,n,decays), [0,tf], y0 = IC, method = 'BDF')
        domains = get_periodic_domains(sol.y,ode.theta_from_param(erratic_param_2),num_periods_to_verify=2)
        check_sequence(domains,[(1,0,0),(1,0,1),(1,1,1),(1,1,0)])


def check_sequence(sequence,expected):
    assert(len(sequence) == len(expected))
    sequence = [tuple(coord) for coord in sequence]
    expected = [tuple(coord) for coord in expected]
    cur_index = expected.index(sequence[0])
    for j in range(len(sequence)):
        assert(sequence[j] == expected[cur_index])
        if cur_index < len(sequence) - 1:
            cur_index += 1
        else:
            cur_index = 0

def repressilator(t,y):
    L = .5
    Delta = 1
    theta = 1
    n = 100
    #repressing hill function
    H = lambda x: L + Delta*(theta**n)/(theta**n + x**n)
    return np.array([-y[0] + H(y[2]), -y[1] + H(y[0]), -y[2] + H(y[1])])
        
