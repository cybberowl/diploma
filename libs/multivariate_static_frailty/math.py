import numpy as np
from numba import njit
from scipy import stats
import math

from ..single_static_frailty.math import *

@njit
def bivariate_density_FG(y1,y2,a1,b1,a2,b2,gamma,F,G1,G2,w=0.5):

    '''
    y1, y2 - result of np.meshgrid
    a1, b1 - parameters of Gompertz distribution for the first spouse
    a2, b2 - parameters of Gompertz distribution for the second spouse
    gamma - parameter from Freund's model
    F, G1, G2 - frailty variables 
    w - weight of the common frailty in personal frailties
    '''
    
    assert(y1.shape == y2.shape)
    
    res1 = np.zeros(y1.shape)
    res2 = np.zeros(y2.shape)
    res = np.zeros(y1.shape)
       
    ## y1 < y2
    res1 = gompertz_mortality(y1,a1,b1)*gompertz_survival_function(y1,a1,b1)**(w*F+(1-w)*G1)*\
    gompertz_survival_function(y1,a2,b2)**(w*F+(1-w)*G2)*gompertz_mortality_conditional(y2 -            y1,y1,a2,b2,gamma)*gompertz_survival_function_conditional(y2-y1,y1,a2,b2,gamma)**G2 * (w*F+(1-w)*G1)*G2
    
    ## y2 < y1
    
    res2 = gompertz_mortality(y2,a2,b2)*gompertz_survival_function(y2,a1,b1)**(w*F+(1-w)*G1)*\
    gompertz_survival_function(y2,a2,b2)**(w*F+(1-w)*G2)*gompertz_mortality_conditional(y1-y2,y2,a1,b1,gamma)*gompertz_survival_function_conditional(y1-y2,y2,a1,b1,gamma)**G1 * (w*F+(1-w)*G2)*G1
        
    res = res1 * (y1<=y2) + res2*(y2<y1)
        
    return res

@njit
def joint_survival_function_FG(x1,x2,a1,b1,a2,b2,F,G1,G2,w=0.5):

    """Joint survival function for the bivariate Gompertz distribution 
    conditional on the frailty variables F, G1, G2.
    
    Parameters
    ----------
    x1 : array-like
        grid of values for the age of the first spouse
    x2 : array-like
        grid of values for the age of the second spouse
    a1 : float
        Gompertz scale parameter for the first spouse
    b1 : float
        Gompertz shape parameter for the first spouse
    a2 : float
        Gompertz scale parameter for the second spouse
    b2 : float
        Gompertz shape parameter for the second spouse
    F, G1, G2 : float
        frailty variables
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    """
    
    return gompertz_survival_function(x1,a1, b1)**(w*F+(1-w)*G1)*\
        gompertz_survival_function(x2,a2, b2)**(w*F+(1-w)*G2)

@njit
def bivariate_density_conditional_FG(y1,y2,z0,a1,b1,a2,b2,gamma,F,G1,G2,w=0.5):

    """Conditional density of mortality with given frailty variables F, G1, G2 
    given the survival up to age z0.
    Parameters
    ----------
    y1 : array-like
        grid of values for the age of the first spouse
    y2 : array-like
        grid of values for the age of the second spouse
    z0 : float
        age up to which spouses have survived
    a1 : float
        Gompertz scale parameter for the first spouse
    b1 : float
        Gompertz shape parameter for the first spouse
    a2 : float
        Gompertz scale parameter for the second spouse
    b2 : float
        Gompertz shape parameter for the second spouse
    gamma : float
        parameter from Freund's model
    F, G1, G2 : float
        frailty variables
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    """
    
    return bivariate_density_FG(y1+z0,y2+z0,a1,b1,a2,b2,gamma,F,G1,G2,w)\
        /joint_survival_function_FG(z0,z0,a1,b1,a2,b2,F,G1,G2,w)


@njit
def bivariate_density_conditional_ndim(y1,y2,z0,a1,b1,a2,b2,gamma,alphaF,thetaF,alphaG_1,thetaG_1,alphaG_2,thetaG_2,w = 0.5):
    
     
    '''
    Integral of bivariate_density_conditional_FG over
    independent gamma triplets with given parameters

    Parameters
    ----------
    y1 : array-like
        grid of values for the age of the first spouse
    y2 : array-like
        grid of values for the age of the second spouse
    z0 : float
        age up to which spouses have survived
    a1 : float
        Gompertz scale parameter for the first spouse
    b1 : float
        Gompertz shape parameter for the first spouse
    a2 : float
        Gompertz scale parameter for the second spouse
    b2 : float
        Gompertz shape parameter for the second spouse  
    gamma : float
        parameter from Freund's model
    alphaF : float
        shape parameter of the gamma distribution for the common frailty
    thetaF : float
        scale parameter of the gamma distribution for the common frailty
    alphaG_1 : float
        shape parameter of the gamma distribution for the first spouse's frailty
    thetaG_1 : float
        scale parameter of the gamma distribution for the first spouse's frailty
    alphaG_2 : float
        shape parameter of the gamma distribution for the second spouse's frailty
    thetaG_2 : float
        scale parameter of the gamma distribution for the second spouse's frailty
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    '''
    
    assert(y1.shape == y2.shape)
    
    ## y1 < y2
    
    thetaF_new = 1/(1/thetaF + w*gompertz_integrated_mortality(y1+z0,a1,b1)+
                    w*gompertz_integrated_mortality(y1+z0,a2,b2) - 
                    w*(gompertz_integrated_mortality(z0,a1,b1)+
                       gompertz_integrated_mortality(z0,a2,b2)))
    
    
    thetaG_1_new = 1/(1/thetaG_1 + (1-w)*gompertz_integrated_mortality(y1+z0,a1,b1) -  
                    (1-w)*gompertz_integrated_mortality(z0,a1,b1))
    
    thetaG_2_new = 1/(1/thetaG_2 + (1-w)*gompertz_integrated_mortality(y1+z0,a2,b2) +
                      gamma * (gompertz_integrated_mortality(y2+z0,a2,b2)-
                              gompertz_integrated_mortality(y1+z0,a2,b2)) -
                    (1-w)*gompertz_integrated_mortality(z0,a2,b2))
    
    res1 = gompertz_mortality(y1+z0,a1,b1)*gompertz_mortality_conditional(y2-y1,y1+z0,a2,b2,gamma)*\
        (thetaF_new/thetaF)**alphaF * (thetaG_1_new/thetaG_1)**alphaG_1 * (thetaG_2_new / thetaG_2)**alphaG_2*\
        alphaG_2 * thetaG_2_new * (w *alphaF * thetaF_new +(1-w)*alphaG_1*thetaG_1_new)
    
    
    ## y2 < y1
    
    thetaF_new = 1/(1/thetaF + w*gompertz_integrated_mortality(y2+z0,a1,b1)+
                    w*gompertz_integrated_mortality(y2+z0,a2,b2) - 
                    w*(gompertz_integrated_mortality(z0,a1,b1)+
                       gompertz_integrated_mortality(z0,a2,b2)))
    
    thetaG_1_new = 1/(1/thetaG_1 + (1-w)*gompertz_integrated_mortality(y2+z0,a1,b1) +
                      gamma * (gompertz_integrated_mortality(y1+z0,a1,b1)-
                              gompertz_integrated_mortality(y2+z0,a1,b1)) -
                    (1-w)*gompertz_integrated_mortality(z0,a1,b1))
    
    thetaG_2_new = 1/(1/thetaG_2 + (1-w)*gompertz_integrated_mortality(y2+z0,a2,b2) -  
                    (1-w)*gompertz_integrated_mortality(z0,a2,b2))
    
    res2 = gompertz_mortality(y2+z0,a2,b2)*gompertz_mortality_conditional(y1-y2,y2+z0,a1,b1,gamma)*\
        (thetaF_new/thetaF)**alphaF * (thetaG_1_new/thetaG_1)**alphaG_1 * (thetaG_2_new / thetaG_2)**alphaG_2*\
        alphaG_1 * thetaG_1_new * (w *alphaF * thetaF_new +(1 - w)*alphaG_2*thetaG_2_new)
    
    
    res = res1 * (y1<=y2) + res2*(y2<y1)
        
    return res

@njit
def n_choose_k(n,k):
    """Compute the binomial coefficient "n choose k"."""
    return math.gamma(n+1)/math.gamma(k+1)/math.gamma(n-k+1)

@njit
def compute_gamma_weights(kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,N1,N2,lambda_,T, w = 0.5, thresh = 0):
    """
    Compute the weights for the mixture based on the parameters and total cases.
    
    Parameters
    ----------
    kF, thetaF : float
        shape and scale parameter for F
    thetaG_1, kG_1 : float
        shape and scale parameter for G1
    thetaG_2, kG_2 : float
        shape and scale parameter for G2
    N1, N2 : int
        total number of cases for first and second spouse
    lambda_ : float
        intensity of the Poisson process
    T : float
        time period of interest
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    thresh : float, optional
        threshold for weights, by default 0    
    """
    weights = np.zeros((N1+1,N2+1))
    
    for n1 in range(0,N1+1):
        for n2 in range(0,N2+1):
            
            weights[n1,n2] = math.gamma(kF+N1+N2-n1-n2)*math.gamma(kG_1+n1)*math.gamma(kG_2+n2)*\
                n_choose_k(N1,n1)*n_choose_k(N2,n2)*((1-w)*(1/thetaF + 2*lambda_*T*w)/w)**(n1+n2)/\
                (1/thetaG_1 + lambda_*T*(1-w))**n1/(1/thetaG_2 + lambda_*T*(1-w))**n2
            
    weights = weights/weights.sum()
    
    for n1 in range(0,N1+1):
        for n2 in range(0,N2+1):
            if weights[n1,n2] < thresh/(N1+1)/(N2+1):
                weights[n1,n2] = 0
                
    weights = weights/weights.sum()
    
    return weights

@njit
def bivariate_density_conditional_data_ndim(y1,y2,z0,a1,b1,a2,b2,gamma,kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,
                                            total_cases1,total_cases2,lambda_,T, w = 0.5):
    
    '''
    Compute the bivariate density function conditional on the data given survival up to age z0.

    Parameters
    ----------  
    y1 : array-like
        grid of values for the age of the first spouse
    y2 : array-like
        grid of values for the age of the second spouse
    z0 : float
        age up to which spouses have survived
    a1 : float
        Gompertz scale parameter for the first spouse       
    b1 : float
        Gompertz shape parameter for the first spouse
    a2 : float
        Gompertz scale parameter for the second spouse
    b2 : float
        Gompertz shape parameter for the second spouse
    gamma : float
        parameter from Freund's model
    kF : float
        shape parameter of the gamma distribution for the common frailty
    thetaF : float
        scale parameter of the gamma distribution for the common frailty
    kG_1 : float
        shape parameter of the gamma distribution for the first spouse's frailty
    thetaG_1 : float
        scale parameter of the gamma distribution for the first spouse's frailty
    kG_2 : float
        shape parameter of the gamma distribution for the second spouse's frailty
    thetaG_2 : float
        scale parameter of the gamma distribution for the second spouse's frailty
    total_cases1 : int
        total number of cases for the first spouse
    total_cases2 : int
        total number of cases for the second spouse
    lambda_ : float
        intensity of the Poisson process
    T : float
        time period of interest
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    '''
     
    thetaF = 1/(1/thetaF + 2*lambda_*T*w)
    thetaG_1 = 1/(1/thetaG_1 + lambda_*T*(1-w))
    thetaG_2 = 1/(1/thetaG_2 + lambda_*T*(1-w))
    
    weights  = compute_gamma_weights(kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,total_cases1,total_cases2,lambda_,T, w)
    
    density = np.zeros_like(y1)
    
    for n1 in range(0,total_cases1+1):
        for n2 in range(0,total_cases2+1):
            if weights[n1,n2]>0:
                density += weights[n1,n2] * bivariate_density_conditional_ndim(
                    y1,y2,z0,a1,b1,a2,b2,gamma,kF + total_cases1 + total_cases2 - n1 - n2,
                    thetaF,kG_1 + n1,thetaG_1,kG_2+n2,thetaG_2,w)
        
    return density

def generate_cases_ndim(z0,lambda_,F,G1,G2,w=0.5,seed = 0):

    """
    Generate cases for two spouses based on the Poisson process and the frailty variables.
    Parameters
    ----------
    z0 : int
        age up to which spouses have survived
    lambda_ : float
        intensity of the Poisson process
    F, G1, G2 : float
        frailty variables for the common and personal frailties
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    seed : int, optional
        random seed for reproducibility, by default 0"""
    
    np.random.seed(seed=seed)
    cases1 = stats.poisson(mu = lambda_*(w*F+(1-w)*G1)).rvs(z0)
    cases2 = stats.poisson(mu = lambda_*(w*F+(1-w)*G2)).rvs(z0)
        
    return cases1, cases2
    
def posterior_gamma_ndim(f,g1,g2,total_cases1,total_cases2,T,lambda_,
                         kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,w = 0.5):
    
    '''
    Compute the posterior distribution of three frailty variables F, G1, G2
    given the data and the parameters of the gamma distributions.
    Parameters
    ----------
    f : array-like
        grid of values for the common frailty variable F
    g1 : array-like
        grid of values for the first spouse's frailty variable G1
    g2 : array-like
        grid of values for the second spouse's frailty variable G2
    total_cases1 : int
        total number of cases for the first spouse
    total_cases2 : int
        total number of cases for the second spouse
    T : float
        time period of interest
    lambda_ : float
        intensity of the Poisson process
    kF : float
        shape parameter of the gamma distribution for the common frailty
    thetaF : float
        scale parameter of the gamma distribution for the common frailty
    kG_1 : float
        shape parameter of the gamma distribution for the first spouse's frailty
    thetaG_1 : float
        scale parameter of the gamma distribution for the first spouse's frailty
    kG_2 : float
        shape parameter of the gamma distribution for the second spouse's frailty
    thetaG_2 : float
        scale parameter of the gamma distribution for the second spouse's frailty
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    '''
        
    thetaF = 1/(1/thetaF + 2*lambda_*T*w)
    thetaG_1 = 1/(1/thetaG_1 + lambda_*T*(1-w))
    thetaG_2 = 1/(1/thetaG_2 + lambda_*T*(1-w))
    
    weights  = compute_gamma_weights(kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,
                                     total_cases1,total_cases2,lambda_,T, w)
    
    density = np.zeros((len(f),len(g1),len(g2)))
                       
    for n1 in range(0,total_cases1+1):
        for n2 in range(0,total_cases2+1):
            if weights[n1,n2]>0:
                gamma_f = stats.gamma.pdf(f,a = kF + total_cases1 + total_cases2 - n1 - n2,
                                  scale = thetaF)
                gamma_g1 = stats.gamma.pdf(g1,a = kG_1 + n1,scale = thetaG_1)
                gamma_g2 = stats.gamma.pdf(g2,a = kG_2 + n2,scale = thetaG_2)

                density += weights[n1,n2]*np.einsum('i,j,k',gamma_f,gamma_g1,gamma_g2)
    
    return density

@njit
def posterior_gamma_moments(total_cases1,total_cases2,T,lambda_,
                         kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,w = 0.5):
    
    """Compute the posterior moments of the gamma distributions for the frailty variables F, G1, G2,
    including the mean, variance, and covariance matrix.
    
    Parameters
    ----------
    total_cases1 : int
        total number of cases for the first spouse
    total_cases2 : int
        total number of cases for the second spouse
    T : float
        time period of interest
    lambda_ : float
        intensity of the Poisson process
    kF : float
        shape parameter of the gamma distribution for the common frailty
    thetaF : float
        scale parameter of the gamma distribution for the common frailty
    kG_1 : float
        shape parameter of the gamma distribution for the first spouse's frailty
    thetaG_1 : float
        scale parameter of the gamma distribution for the first spouse's frailty
    kG_2 : float
        shape parameter of the gamma distribution for the second spouse's frailty
    thetaG_2 : float
        scale parameter of the gamma distribution for the second spouse's frailty
    w : float, optional
        weight of the common frailty in personal frailties, by default 0.5
    """
    
    weights = compute_gamma_weights(kF,thetaF,kG_1,thetaG_1,kG_2,thetaG_2,
                                     total_cases1,total_cases2,lambda_,T, w)
        
    mean = np.zeros(3)
    variance = np.zeros(3)
    cov = np.zeros((3,3))
    
    thetaF = 1/(1/thetaF + 2*lambda_*T*w)
    thetaG_1 = 1/(1/thetaG_1 + lambda_*T*(1-w))
    thetaG_2 = 1/(1/thetaG_2 + lambda_*T*(1-w))
    
    n1 = np.zeros((total_cases1+1,total_cases2+1))
    n2 = np.zeros((total_cases1+1,total_cases2+1))
    
    for n_1 in range(total_cases1+1):
        for n_2 in range(total_cases2 + 1):
            n1[n_1,n_2] = n_1
            n2[n_1,n_2] = n_2
    
    F_mean = (kF + total_cases1 + total_cases2 - n1 - n2)*thetaF
    G_1_mean = (kG_1 + n1)*thetaG_1
    G_2_mean = (kG_2 + n2)*thetaG_2
    
    F_var = (kF + total_cases1 + total_cases2 - n1 - n2)*thetaF**2
    G_1_var = (kG_1 + n1)*thetaG_1**2
    G_2_var = (kG_2 + n2)*thetaG_2**2
    
    ## expectation
    for i,var in enumerate([F_mean,G_1_mean,G_2_mean]):
        mean[i] = (var*weights).sum()
    
    ## conditional variance
    for i,var in enumerate([F_var,G_1_var,G_2_var]):    
        variance[i] = (var*weights).sum()  
    
    ## covariance of conditional means
    for i,var_1 in enumerate([F_mean,G_1_mean,G_2_mean]):
        for j,var_2 in enumerate([F_mean,G_1_mean,G_2_mean]):
            cov[i,j] = ((var_1-mean[i])*(var_2-mean[j])*weights).sum()
    
    ## add conditional variance
    for i in range(cov.shape[0]):
        cov[i,i] += variance[i]    
     
    ## transform variance to std
    for i in range(cov.shape[0]):
        cov[i,i] = np.sqrt(cov[i,i])
         
    ## transform covariance to correlation
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if i!=j:
                cov[i,j] = cov[i,j]/cov[i,i]/cov[j,j]
            
    return cov,mean
    
    
    


