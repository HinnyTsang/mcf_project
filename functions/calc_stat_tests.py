import numpy as np
# from itertools import combinations as comb
import math

def comb(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

# Statistical test for projection only.
def projection_test(n, para, perp):
    """
        :param n: number of projections
        :param para: parallel cloud
        :param perp: perpendicular cloud
        :return: Tuple of parameters.
    """
    
    # store output
    num_pa_c = np.zeros([(n+1), int(n+1)])
    num_pa_pj = np.zeros([(n+1), int(n+1)])
    prob = np.zeros([(n+1), int(n+1)])
    
    # calculate probability of observating parallel projections.
    prob_pac_paj = np.sum(para['b_offset'] < 45) / para['b_offset'].size
    prob_pac_pej = np.sum(para['b_offset'] > 45) / para['b_offset'].size
    prob_pec_paj = np.sum(perp['b_offset'] < 45) / perp['b_offset'].size
    prob_pec_pej = np.sum(perp['b_offset'] > 45) / perp['b_offset'].size    

    print(prob_pac_paj, prob_pac_pej)

    # number of parallel cloud
    for k in range(n+1): 
        # number of parallel projection
        for r in  range(n+1):
            
            
            num_pa_c[k][r] = k
            num_pa_pj[k][r] = r
            # number of parallel projection from parallel cloud
            for m in range(n+1):
                if n-k-r+m < 0 or r-m < 0 or k-m < 0:
                    continue
                curr = comb(int(k), int(m))*prob_pac_paj**m*prob_pac_pej**(k - m)
                curr*= comb(int(n-k), int(r-m))*prob_pec_paj**(r-m)*prob_pec_pej**(n-k-r+m)
                prob[k][r] += curr
                
    mean_n_pa = []
    std_n_pa  = []

    for k in range(n+1):
        mean_n_pa += [np.sum(prob[k]*num_pa_pj[k])]
        std_n_pa += [np.sqrt(np.sum(prob[k]*num_pa_pj[k]**2) - mean_n_pa[-1]**2)]

    return prob, num_pa_c, num_pa_pj, np.array(mean_n_pa), np.array(std_n_pa)