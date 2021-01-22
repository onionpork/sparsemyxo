import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import itertools, operator, random, math
from scipy.sparse.linalg import spsolve_triangular
from sklearn import linear_model
import pandas as pd

def random_sampling(data, porpotion):
    sampled_data = np.empty(data.shape)
    sampled_data[:] = np.nan
    n = data.shape[1]
    for i in range(data.shape[0]):
        sample_idx = random.sample(range(n), int(n*porpotion))
        sampled_data[i][sample_idx] = data[i][sample_idx]
    return sampled_data

def funkSVD(rating_mat, latent_features, learning_rate, iters):
    n_s, n_t = rating_mat.shape[0], rating_mat.shape[1]
    s_matrix, t_matrix = np.random.rand(n_s, latent_features), np.random.rand(latent_features, n_t)
    # s_matrix, t_matrix = 0.5*np.ones((n_s, latent_features)), 0.5*np.ones((latent_features, n_t))
    sse_initial = 0
    for p in range(iters):
        old_see = sse_initial
        sse_initial = 0 

        for i in range(n_s):
            for j in range(n_t):
                if not math.isnan(rating_mat[i][j]):
                    diff = rating_mat[i][j] - s_matrix[i,:].dot(t_matrix[:,j])
                    sse_initial  += diff**2

                    for k in range(latent_features):
                        s_matrix[i][k] += learning_rate*(2*diff*t_matrix[k][j])
                        t_matrix[k][j] += learning_rate*(2*diff*s_matrix[i][k])

    est_mat = s_matrix.dot(t_matrix)
    return est_mat


def ft_data(pop, tspan, dt):
    """
    est_mat from funkSVD
    """
    n = len(tspan)
    y_ft = []
    for i in range(pop.shape[0]):
        fhat = np.fft.fft(pop[i], n)
        PSD = fhat*np.conj(fhat)/n
        freq = (1/(dt*n))*np.arange(n)

        L = np.arange(1, np.floor(n/2), dtype= 'int')

        indices = PSD > 5
        PSDclean  = PSD * indices
        fhat = indices*fhat
        ffilt = np.fft.ifft(fhat)
        y_ft.append(ffilt)
    return np.array(y_ft)


def funkSVD_ft(ft_matrix, rating_mat, latent_features, learning_rate, iters):
    
    u,s,v = np.linalg.svd(ft_matrix, full_matrices=False)
    n_s, n_t = rating_mat.shape[0], rating_mat.shape[1]
    s_matrix, t_matrix = u, v
    # s_matrix, t_matrix = 0.5*np.ones((n_s, latent_features)), 0.5*np.ones((latent_features, n_t))
    sse_initial = 0
    for p in range(iters):
        old_see = sse_initial
        sse_initial = 0 

        for i in range(n_s):
            for j in range(n_t):
                if not math.isnan(rating_mat[i][j]):
                    diff = rating_mat[i][j] - s_matrix[i,:].dot(t_matrix[:,j])
                    sse_initial  += diff**2

                    for k in range(latent_features):
                        s_matrix[i][k] += learning_rate*(2*diff*t_matrix[k][j])
                        t_matrix[k][j] += learning_rate*(2*diff*s_matrix[i][k])

    est_mat = s_matrix.dot(t_matrix)
    return est_mat


def power_(d,order):
# d is the number of variables; order of polynomials
    powers = []
    for p in range(1,order+1):
        size = d + p - 1
        for indices in itertools.combinations(range(size), d-1):   ##combinations
            starts = [0] + [index+1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
    return powers
def lib_terms(data,order,description):
    #description is a list of name of variables, like [R, M, S]
    #description of lib
    descr = []
    #data is the input data, like R,M,S; order is the total order of polynomials

    d,t = data.shape # d is the number of variables; t is the number of time points
    theta = np.ones((t,1), dtype=np.float64) # the first column of lib is '1'
    P = power_(d,order)
    descr = ["1"]
    for i in range(len(P)):
        new_col = np.zeros((t,1),dtype=np.float64)
        for j in range(t):
            new_col[j] = np.prod(np.power(list(data[:,j]),list(P[i])))
        theta = np.hstack([theta, new_col.reshape(t,1)])
        descr.append("{0} {1}".format(str(P[i]), str(description)))
        # print((str(P[i]), str(description)))
    

    return theta, descr



def sparsifyDynamics(Theta, dx, Lambda):
    #theta.shape = 248*10 (time points*functions); dx.shape = 248*3 (time points*variables)
    #need to ensure size or dimenssions !!!
#     dx = dx.T
    m,n = dx.shape  #(248*3)
    Xi = np.dot(np.linalg.pinv(Theta), dx)  #Xi.shape = 10*3
    # lambda is sparasification knob
    for k in range(20):      ###??
        small_idx = (abs(Xi) < Lambda)
        big_idx = (abs(Xi) >= Lambda)
        Xi[small_idx] = 0
        for i in range(n):
            big_curr, = np.where(big_idx[:,i])
            Xi[big_curr, i] = np.dot(np.linalg.pinv(Theta[:,big_curr]), dx[:,i])
    return Xi 


def sparseGalerkin(t, pop, Xi, polyorder):
    theta, descr = lib_terms(np.array([pop]).T,polyorder,[])
    dpop = theta.dot(Xi)
    return dpop[0]


def time_different(dt, pop):
    """
    dpop = (6*6000) (species * time)
    centered first order derviate
    """
    x = np.full_like(pop, fill_value = np.nan)
    x[:, 1:-1] = (pop[:, 2:] - pop[:, :-2]) / (2*dt)
    x[:,0] = (-11/6 *pop[:,0] + 3* pop[:,1] - 3/2*pop[:,2] + pop[:,3]/3) /dt
    x[:,-1] = (11/6* pop[:,-1] -3* pop[:,-2] + 3/2* pop[:,-3] -pop[:,-4]/3)/dt

    return x


def visual_param(Xi, descr):
    small_idx = abs(Xi) < 1e-4
    Xi[small_idx] = 0
    new_set =  [x.replace('(', '').replace(']', '') for x in descr] 
    name_s = descr
    label = []
    for str_ in new_set[1:]:
        idx_ = [int(x) for x in str_.split(') [')[0].split(',')]
        lab = ""
        for idx, i in enumerate(idx_):
            j = i
            while j > 0:
                lab += name_s[idx]
                j -= 1
        label.append(lab)

    term_label = ['1'] + label

    df_term = pd.DataFrame(Xi.T, index=term_label, columns=name_s)

    return df_term



def bulid_prior(label, theta, descr, prior_dic):

    df_prior = visual_param(np.zeros((len(label), theta.shape[1])), descr)
    drop_index = []

    for term in label:
        idx_prev =  df_prior.index
        x_new = set()
        for i, s in enumerate(prior_dic[term]):
            lst_idx = [p.find(s) for p in idx_prev]
            x, = np.where(np.array(lst_idx) == -1)
            if i == 0:
                x_new = set(x)
            else:
                x_new = x_new.intersection(x)
        drop_index.append(list(x_new))
        df_prior[term].iloc[list(x_new)] = 1
    return df_prior, drop_index
