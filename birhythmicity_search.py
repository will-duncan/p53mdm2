'''
Create a list of parameter pairs for each parameter node. The line connecting
the parameter pairs contains candidate parameters for birhythmicity which can 
be manually checked. 
'''


from ode_functions import param_line,generate_all_lines, check_line_list,make_parameters
import json
import time

if __name__ == '__main__':
    param_labels = ['E1','M1F','M1R','MB1R','B1R','TM1F','T1F']
    with open('results.json','r') as reader:
        classification_results = json.load(reader)
    lines_to_check = {label:[] for label in param_labels}
    params_sampled_per_label = 100
    num_params_per_line = 20
    for label in param_labels:
        tic = time.time()
        print('Searching ' + label, flush = True)
        params = make_parameters(label,params_sampled_per_label)
        large_indices = classification_results[label]['large_params']
        small_indices = classification_results[label]['small_params']
        for s_i in small_indices:
            if s_i < 50:
                for L_i in large_indices:
                    if L_i < 50:
                        cur_line = param_line(params[s_i],params[L_i],num_params_per_line)
                        is_interesting = check_line_list([cur_line])[0]
                        if is_interesting:
                            lines_to_check[label].append([s_i,L_i])
                            print('Found param combo to check: ' + str(s_i) + ' ' +  str(L_i),flush = True)
        toc = time.time()
        time_lapsed = toc - tic
        print(label + 'done in ' + str(time_lapsed) + 'sec.',flush = True)
    with open('birhythmicity_results.json','w') as writer:
        json.dump(lines_to_check,writer)

