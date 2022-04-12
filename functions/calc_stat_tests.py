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


# Statistical test for mcf slope
def mcf_slope_test(n, r, para, perp, sampling):
    
    np.random.seed(0)
    
    mcf_pa = para['mcf_slope']
    mcf_pe = perp['mcf_slope']
    
    bof_pa = para['b_offset']
    bof_pe = perp['b_offset']

    mcf_pe_pa = mcf_pa[bof_pa > 45]
    bof_pe_pa = bof_pa[bof_pa > 45]
    
    
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


# Statistical test for dense gas fraction
def dense_gass_fraction_test(n, r, para, perp, sampling, dgf_para, dgf_perp):
    
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


