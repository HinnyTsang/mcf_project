"""
    Different statistical tests
    
    Author: Hinny Tsang
    Last Edit: 2022-04-13
"""

from typing import Dict, List, Tuple
import numpy as np
# from itertools import combinations as comb
import math


def normal(x:np.ndarray, mu: float, sig:float)->np.ndarray:
    """
        Normal distribution.
        :param x:   x
        :param mu:  mean of the gaussian
        :param sig: sd of the gaussian
    """
    N = 1/np.sqrt(2*np.pi)
    N /= sig
    
    X = (x - mu)/sig
    X **= 2
    X /= -2
    
    return N * np.exp(X)


def comb(n: int,r: int) -> int:
    f = math.factorial
    return f(n) / f(r) / f(n-r)

# Statistical test for projection only.
def projection_test(n:int , b_offset_para: np.ndarray, b_offset_perp: np.ndarray) -> Dict:
    """
        :param n:               number of projections
        :param b_offset_para:   parallel cloud
        :param b_offset_perp:   perpendicular cloud
        :return:                Dictionary of parameters
    """
    
    # store output
    num_pa_c = np.zeros([(n+1), int(n+1)])
    num_pa_pj = np.zeros([(n+1), int(n+1)])
    prob = np.zeros([(n+1), int(n+1)])
    
    # calculate probability of observating parallel projections.
    prob_pac_paj = np.sum(b_offset_para < 45) / b_offset_para.size
    prob_pac_pej = np.sum(b_offset_para > 45) / b_offset_para.size
    prob_pec_paj = np.sum(b_offset_perp < 45) / b_offset_perp.size
    prob_pec_pej = np.sum(b_offset_perp > 45) / b_offset_perp.size    

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


    return {
        'probability': prob, 
        'n_para_cloud': num_pa_c, 
        'n_para_proj': num_pa_pj
    }

# Statistical test for any parameters
def parameter_test_given_r(
    n: int, r: int,
    b_offset_para: np.ndarray,
    b_offset_perp: np.ndarray,
    parameter_para: np.ndarray,
    parameter_perp: np.ndarray,
    sampling: int) -> List:
    """
        :param n:               number of projections total
        :param r:               number of parallel projections
        :param b_offset_para:   cloud field offset of parallel cloud 
        :param b_offset_perp:   cloud field offset of perpendcular cloud
        :param parameter_para:  any parameters of parallel cloud 
        :param parameter_perp:  any parameters of perpendcular cloud
        :param sampling:        number of boostrapping.
    """
    np.random.seed(0)

    parameter_perp_proj_para_cloud = np.array(parameter_para[b_offset_para > 45])
    b_offset_perp_proj_para_cloud = np.array(b_offset_para[b_offset_para > 45])
    
    result = []
    
    # k is number of parallel cloud
    for k in range(n+1):
        
        # number of parallel cloud gives perpendicular projection
        o = max(k-r, 0) 

        # contour for successful test.
        sample_size = 0
        
        # store each boostrapping data.
        curr = []
        
        # weighted mean of parameter
        parameter_weighted_mean = (k/n)*np.mean(parameter_para) + ((n-k)/n)*np.mean(parameter_perp)
        
        
        while sample_size < sampling:
            
            # Impossible to have perpendicular projection form parallel cloud.
            if o > 0 and parameter_perp_proj_para_cloud.size == 0:
                curr += [np.nan]
                break
            
            # randomly pick o perpendicular projection from parallel cloud.
            idx_pe_pa = np.random.choice(parameter_perp_proj_para_cloud.size, o, replace = True)
            # randomly pick k-o projections from the parallel cloud
            idx_pa = np.random.choice(parameter_para.size, k-o, replace = True)
            # randomly pick n-k projections from the parallel cloud
            idx_pe = np.random.choice(parameter_perp.size, n-k, replace = True)
            
            # ensure number of parallel projection is r
            n_para = np.sum(b_offset_para[idx_pa] < 45) + np.sum(b_offset_perp[idx_pe] < 45)
            if (n_para!= r):
                continue
            
            param = np.concatenate([parameter_perp_proj_para_cloud[idx_pe_pa], parameter_para[idx_pa], parameter_perp[idx_pe]])
            bof = np.concatenate([b_offset_perp_proj_para_cloud[idx_pe_pa], b_offset_para[idx_pa], b_offset_perp[idx_pe]])
            
            # calculate the relative mcf slope different
            parameter_para_proj_mean = np.mean(param[bof < 45])/parameter_weighted_mean
            parameter_perp_proj_mean = np.mean(param[bof > 45])/parameter_weighted_mean
            
            curr += [parameter_perp_proj_mean - parameter_para_proj_mean]
            sample_size += 1


        result += [np.array(curr)]
        
    return {'result': np.array(result)}    

# Statistical test for any parameters
def parameter_test(
    n: int,
    b_offset_para: np.ndarray,
    b_offset_perp: np.ndarray,
    parameter_para: np.ndarray,
    parameter_perp: np.ndarray,
    sampling: int) -> List:
    """
        :param n:               number of projections total
        :param r:               number of parallel projections
        :param b_offset_para:   cloud field offset of parallel cloud 
        :param b_offset_perp:   cloud field offset of perpendcular cloud
        :param parameter_para:  any parameters of para cloud 
        :param parameter_perp:  any parameters of perpendcular cloud
        :param sampling:        number of boostrapping.
    """
    np.random.seed(0)

    result = []
    n_para_proj = [] # number of parallel projections.
    
    # k is number of parallel cloud
    for k in range(n+1):
        
        # contour for successful test.
        sample_size = 0
        
        # store each boostrapping data.
        curr = []
        curr_n_para_proj = []
        
        # weighted mean of paramter
        paramter_weighted_mean = (k/n)*np.mean(parameter_para) + ((n-k)/n)*np.mean(parameter_perp)
        
        
        while sample_size < sampling:
            
            # randomly pick k-o projections from the parallel cloud
            idx_pa = np.random.choice(parameter_para.size, k, replace = True)
            # randomly pick n-k projections from the parallel cloud
            idx_pe = np.random.choice(parameter_perp.size, n-k, replace = True)
            
            # TODO calculate the relative mcf slope different #####################
            mcf = np.concatenate([parameter_para[idx_pa], parameter_perp[idx_pe]])
            bof = np.concatenate([b_offset_para[idx_pa], b_offset_perp[idx_pe]])
            
            # if all cloud ara parallel or all cloud are perpendicular, mcf slope different is not defined
            if np.sum(bof < 45) == 0 or np.sum(bof > 45) == 0:
                curr += [np.nan]

            else:
                paramter_para_proj_mean = np.mean(mcf[bof < 45])/paramter_weighted_mean
                paramter_perp_proj_mean = np.mean(mcf[bof > 45])/paramter_weighted_mean
                curr += [paramter_perp_proj_mean - paramter_para_proj_mean]
            ######################################################################
            
            # store the values.
            curr_n_para_proj += [np.sum(bof < 45)]
            sample_size += 1

        result += [np.array(curr)]
        n_para_proj += [np.array(curr_n_para_proj)]
        
    return {'result':np.array(result),
            'n_para': np.array(n_para_proj)}















# The three tests below are not usefull.

# Statistical test for mcf slope
def mcf_slope_test_given_r(
    n: int, r: int,
    b_offset_para: np.ndarray,
    b_offset_perp: np.ndarray,
    mcf_slope_para: np.ndarray,
    mcf_slope_perp: np.ndarray,
    sampling: int) -> List:
    """
        :param n:               number of projections total
        :param r:               number of parallel projections
        :param b_offset_para:   cloud field offset of parallel cloud 
        :param b_offset_perp:   cloud field offset of perpendcular cloud
        :param mcf_slope_para:  mcf slope of parallel cloud 
        :param mcf_slope_perp:  mcf slope of perpendcular cloud
        :param sampling:        number of boostrapping.
    """
    np.random.seed(0)

    mcf_slope_perp_proj_para_cloud = mcf_slope_para [b_offset_para > 45]
    b_offset_perp_proj_para_cloud = b_offset_para [b_offset_para > 45]
    
    result = []
    
    # k is number of parallel cloud
    for k in range(n+1):
        
        # number of parallel cloud gives perpendicular projection
        o = max(k-r, 0) 

        # contour for successful test.
        sample_size = 0
        
        # store each boostrapping data.
        curr = []
        
        # weighted mean of mcf_slope
        mcf_slope_weighted_mean = (k/n)*np.mean(mcf_slope_para) + ((n-k)/n)*np.mean(mcf_slope_perp)
        
        
        while sample_size < sampling:
            
            # Impossible to have perpendicular projection form parallel cloud.
            if o > 0 and mcf_slope_perp_proj_para_cloud.size == 0:
                curr += [np.nan]
                break
            
            # randomly pick o perpendicular projection from parallel cloud.
            idx_pe_pa = np.random.choice(mcf_slope_perp_proj_para_cloud.size, o, replace = True)
            # randomly pick k-o projections from the parallel cloud
            idx_pa = np.random.choice(mcf_slope_para.size, k-o, replace = True)
            # randomly pick n-k projections from the parallel cloud
            idx_pe = np.random.choice(mcf_slope_perp.size, n-k, replace = True)
            
            # ensure number of parallel projection is r
            n_para = np.sum(b_offset_para[idx_pa] < 45) + np.sum(b_offset_perp[idx_pe] < 45)
            if (n_para!= r):
                continue
            
            mcf = np.concatenate([mcf_slope_perp_proj_para_cloud[idx_pe_pa], mcf_slope_para[idx_pa], mcf_slope_perp[idx_pe]])
            bof = np.concatenate([b_offset_perp_proj_para_cloud[idx_pe_pa], b_offset_para[idx_pa], b_offset_perp[idx_pe]])
            
            # calculate the relative mcf slope different
            mcf_slope_para_proj_mean = np.mean(mcf[bof < 45])/mcf_slope_weighted_mean
            mcf_slope_perp_proj_mean = np.mean(mcf[bof > 45])/mcf_slope_weighted_mean
            
            curr += [mcf_slope_perp_proj_mean - mcf_slope_para_proj_mean]
            sample_size += 1


        result += [np.array(curr)]
        
    return {'result': np.array(result)}    

# Statistical test for dense gas fraction
def dense_gass_fraction_test_given_r(n, r, para, perp, sampling, dgf_para, dgf_perp):
    
    np.random.seed(0)
    
    mcf_pa = dgf_para
    mcf_pe = dgf_perp
    
    bof_pa = para['b_offset']
    bof_pe = perp['b_offset']

    
    bof_pe_pa = bof_pa[bof_pa > 45]
    mcf_pe_pa = np.array([mcf_pa]*bof_pe_pa.size) # [bof_pa > 45]
    
    
    result = []
    
    for k in range(n+1):
        
        o = max(k-r, 0) # number of parallel cloud gives perpendicular projection

        sample_size = 0
        
        curr = []
        
        mcf_mean = (k/n)*np.mean(mcf_pa) + ((n-k)/n)*np.mean(mcf_pe)
        
        while sample_size < sampling:
            
            if o > 0 and mcf_pe_pa.size == 0:
                break
            idx_pe_pa = np.random.choice(mcf_pe_pa.size, o, replace = True)
            idx_pa = np.random.choice(mcf_pa.size, k-o, replace = True)
            idx_pe = np.random.choice(mcf_pe.size, n-k, replace = True)
            
            # ensure number of parallel projection is 5
            n_para = np.sum(bof_pa[idx_pa] < 45) + np.sum(bof_pe[idx_pe] < 45)
            if (n_para!= r):
                continue
            
            mcf = np.concatenate([mcf_pe_pa[idx_pe_pa], mcf_pa[idx_pa], mcf_pe[idx_pe]])
            bof = np.concatenate([bof_pe_pa[idx_pe_pa], bof_pa[idx_pa], bof_pe[idx_pe]])
            
            
            mcf_pa_mean = np.mean(mcf[bof < 45])/mcf_mean
            mcf_pe_mean = np.mean(mcf[bof > 45])/mcf_mean
            
            curr += [mcf_pe_mean - mcf_pa_mean]
            sample_size += 1

        result += [np.array(curr)]
        
    return np.array(result)    

# Statistical test for mcf slope
def mcf_slope_test(
    n: int,
    b_offset_para: np.ndarray,
    b_offset_perp: np.ndarray,
    mcf_slope_para: np.ndarray,
    mcf_slope_perp: np.ndarray,
    sampling: int) -> List:
    """
        :param n:               number of projections total
        :param r:               number of parallel projections
        :param b_offset_para:   cloud field offset of parallel cloud 
        :param b_offset_perp:   cloud field offset of perpendcular cloud
        :param mcf_slope_para:  mcf slope of parallel cloud 
        :param mcf_slope_perp:  mcf slope of perpendcular cloud
        :param sampling:        number of boostrapping.
    """
    np.random.seed(0)

    result = []
    n_para_proj = [] # number of parallel projections.
    
    # k is number of parallel cloud
    for k in range(n+1):
        
        # contour for successful test.
        sample_size = 0
        
        # store each boostrapping data.
        curr = []
        curr_n_para_proj = []
        
        # weighted mean of mcf_slope
        mcf_slope_weighted_mean = (k/n)*np.mean(mcf_slope_para) + ((n-k)/n)*np.mean(mcf_slope_perp)
        
        
        while sample_size < sampling:
            
            # randomly pick k-o projections from the parallel cloud
            idx_pa = np.random.choice(mcf_slope_para.size, k, replace = True)
            # randomly pick n-k projections from the parallel cloud
            idx_pe = np.random.choice(mcf_slope_perp, n-k, replace = True)
            
            # ensure number of parallel projection is r
            n_para = np.sum(b_offset_para[idx_pa] < 45) + np.sum(b_offset_perp[idx_pe] < 45)
            if (n_para!= r):
                continue
            
            mcf = np.concatenate([mcf_slope_para[idx_pa], mcf_slope_perp[idx_pe]])
            bof = np.concatenate([b_offset_para[idx_pa], b_offset_perp[idx_pe]])
            
            # calculate the relative mcf slope different
            mcf_slope_para_proj_mean = np.mean(mcf[bof < 45])/mcf_slope_weighted_mean
            mcf_slope_perp_proj_mean = np.mean(mcf[bof > 45])/mcf_slope_weighted_mean
            
            curr += [mcf_slope_perp_proj_mean - mcf_slope_para_proj_mean]
            curr_n_para_proj += [np.sum(bof > 45)]
            sample_size += 1

        result += [np.array(curr)]
        n_para_proj += [np.array(curr_n_para_proj)]
        
    return np.array(result), np.array(n_para_proj)
