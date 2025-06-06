import numpy as np
from numba import njit
from scipy import stats


@njit
def gompertz_mortality(x, alpha, beta):
    """
    Calculates the Gompertz mortality function.

    The Gompertz mortality function is commonly used in actuarial science and demography
    to model the age-specific mortality rate, which increases exponentially with age.

    Parameters:
        x (float or np.ndarray): Age or array of ages at which to evaluate the mortality.
        alpha (float): The rate at which mortality increases with age (shape parameter).
        beta (float): The baseline mortality rate (scale parameter).

    Returns:
        float or np.ndarray: The mortality rate(s) at age x.
    """
    return np.exp(alpha * x + beta)


@njit
def gompertz_integrated_mortality(x, alpha, beta):
    """
    Calculates the integrated Gompertz mortality function.
    The integrated Gompertz mortality function gives the cumulative mortality from age 0 to age x.
    """
    return (np.exp(alpha * x + beta) - np.exp(beta)) / alpha


@njit
def gompertz_mortality_conditional(x, z, alpha, beta, gamma=1):
    """
    Calculates the conditional Gompertz mortality function from Freund model.
    """
    return gamma * gompertz_mortality(x + z, alpha, beta)


@njit
def gompertz_survival_function(x, alpha, beta):
    """
    Calculates the Gompertz survival function.
    The Gompertz survival function gives the probability of surviving to age x."""
    return np.exp(-gompertz_integrated_mortality(x, alpha, beta))


@njit
def gompertz_survival_function_conditional(x, z, alpha, beta, gamma=1):
    """
    Calculates the conditional Gompertz survival function from Freund model.
    The conditional survival function gives the probability of surviving to age x given that
    other spouse died at age z.
    """
    return np.exp(
        -gamma
        * (
            gompertz_integrated_mortality(x + z, alpha, beta)
            - gompertz_integrated_mortality(z, alpha, beta)
        )
    )


@njit
def gompertz_density(x, alpha, beta):
    """
    Calculates the Gompertz density function."""
    return gompertz_mortality(x, alpha, beta) * gompertz_survival_function(
        x, alpha, beta
    )


@njit
def bivariate_density_F(y1, y2, a1, b1, a2, b2, gamma, F):
    """
    y1, y2 - result of np.meshgrid
    a1, b1 - parameters of Gompertz distribution for first spouse
    a2, b2 - parameters of Gompertz distribution for second spouse
    gamma - parameter of Freund model
    F - frailty parameter
    This function calculates the bivariate density function for two spouses conditional on frailty F.
    """

    assert y1.shape == y2.shape

    res1 = np.zeros(y1.shape)
    res2 = np.zeros(y2.shape)
    res = np.zeros(y1.shape)

    ## y1 < y2
    res1 = (
        gompertz_mortality(y1, a1, b1)
        * gompertz_survival_function(y1, a1, b1) ** F
        * gompertz_survival_function(y1, a2, b2) ** F
        * gompertz_mortality_conditional(y2 - y1, y1, a2, b2, gamma)
        * gompertz_survival_function_conditional(y2 - y1, y1, a2, b2, gamma) ** F
    )

    ## y2 < y1

    res2 = (
        gompertz_mortality(y2, a2, b2)
        * gompertz_survival_function(y2, a1, b1) ** F
        * gompertz_survival_function(y2, a2, b2) ** F
        * gompertz_mortality_conditional(y1 - y2, y2, a1, b1, gamma)
        * gompertz_survival_function_conditional(y1 - y2, y2, a1, b1, gamma) ** F
    )

    res = res1 * (y1 <= y2) + res2 * (y2 < y1)

    return res * F**2


@njit
def joint_survival_function_F(x1, x2, a1, b1, a2, b2, F):

    """
    Calculates the joint survival function for two spouses conditional on frailty F.
    
    Parameters:
    x1, x2 : float or np.ndarray
        Ages of the first and second spouse.
    a1, b1 : float
        Parameters of the Gompertz distribution for the first spouse.
    a2, b2 : float
        Parameters of the Gompertz distribution for the second spouse.
    F : float
        Frailty parameter.
    """

    return (
        gompertz_survival_function(x1, a1, b1) * gompertz_survival_function(x2, a2, b2)
    ) ** F


@njit
def bivariate_density_conditional_F(y1, y2, z0, a1, b1, a2, b2, gamma, F):

    """
    This function calculates the bivariate density function
    for two spouses conditional on frailty F given that both spouses survived to age z0.
    Parameters:
    y1, y2 : np.ndarray
        Arrays of ages for the first and second spouse.
    z0 : float
        Age at which both spouses are known to have survived.
    a1, b1 : float
        Parameters of the Gompertz distribution for the first spouse.
    a2, b2 : float
        Parameters of the Gompertz distribution for the second spouse.
    gamma : float
        Parameter of the Freund model.
    F : float
        Frailty parameter.
    """

    return (
        bivariate_density_F(y1 + z0, y2 + z0, a1, b1, a2, b2, gamma, F)
        / np.ones(y1.shape)
        / joint_survival_function_F(z0, z0, a1, b1, a2, b2, F)
    )


@njit
def bivariate_density_conditional(y1, y2, z0, a1, b1, a2, b2, gamma, k, theta):
    """
    This function calculates the bivariate density function
    for two spouses given that both spouses survived to age z0, 
    integrating out the frailty parameter F.

    Parameters:
    y1, y2 : np.ndarray
        Arrays of ages for the first and second spouse.
    z0 : float
        Age at which both spouses are known to have survived.
    a1, b1 : float
        Parameters of the Gompertz distribution for the first spouse.
    a2, b2 : float
        Parameters of the Gompertz distribution for the second spouse.
    gamma : float
        Parameter of the Freund model.
    k : float
        Shape parameter for the gamma distribution of the frailty.
    theta : float
        Scale parameter for the gamma distribution of the frailty.
    """

    assert y1.shape == y2.shape

    res1 = np.zeros(y1.shape)
    res2 = np.zeros(y2.shape)

    ## y1 < y2

    res1 = (
        gompertz_mortality(y1 + z0, a1, b1)
        * gompertz_mortality(y2 + z0, a2, b2)
        * gamma
        * k
        * (k + 1)
        / (
            theta**k
            * (
                1 / theta
                + gompertz_integrated_mortality(y1 + z0, a1, b1)
                + gompertz_integrated_mortality(y1 + z0, a2, b2)
                + gamma * gompertz_integrated_mortality(y2 + z0, a2, b2)
                - gamma * gompertz_integrated_mortality(y1 + z0, a2, b2)
                - gompertz_integrated_mortality(z0, a1, b1)
                - gompertz_integrated_mortality(z0, a2, b2)
            )
            ** (k + 2)
        )
    )

    ## y2 < y1

    res2 = (
        gompertz_mortality(y1 + z0, a1, b1)
        * gompertz_mortality(y2 + z0, a2, b2)
        * gamma
        * k
        * (k + 1)
        / (
            theta**k
            * (
                1 / theta
                + gompertz_integrated_mortality(y2 + z0, a1, b1)
                + gompertz_integrated_mortality(y2 + z0, a2, b2)
                + gamma * gompertz_integrated_mortality(y1 + z0, a1, b1)
                - gamma * gompertz_integrated_mortality(y2 + z0, a1, b1)
                - gompertz_integrated_mortality(z0, a1, b1)
                - gompertz_integrated_mortality(z0, a2, b2)
            )
            ** (k + 2)
        )
    )

    res = res1 * (y1 <= y2) + res2 * (y2 < y1)

    return res


@njit
def bivariate_density_conditional_data(
    y1, y2, z0, a1, b1, a2, b2, gamma, k, theta, total_cases, lambda_, T
):
    """
    This function calculates the bivariate density function
    for two spouses given that both spouses survived to age z0, 
    conditional on the total number of cases observed.

    Parameters:
    y1, y2 : np.ndarray
        Arrays of ages for the first and second spouse.
    z0 : float
        Age at which both spouses are known to have survived.
    a1, b1 : float
        Parameters of the Gompertz distribution for the first spouse.
    a2, b2 : float
        Parameters of the Gompertz distribution for the second spouse.
    gamma : float
        Parameter of the Freund model.
    k : float
        Shape parameter for the gamma distribution of the frailty.
    theta : float
        Scale parameter for the gamma distribution of the frailty.
    total_cases : int
        Total number of cases observed.
    lambda_ : float
        Rate parameter for the Poisson distribution of cases.
    T : float
        Time period over which the cases are observed.

    """

    k = k + total_cases
    theta = 1 / (1 / theta + 2 * lambda_ * T)

    return bivariate_density_conditional(y1, y2, z0, a1, b1, a2, b2, gamma, k, theta)


def generate_cases(z0, lambda_, F, seed=0):

    """
    Generates two sets of cases based on a Poisson distribution with a given lambda and frailty F.
    Parameters:
    z0 : int
        The number of cases to generate.
    lambda_ : float
        The rate parameter for the Poisson distribution.
    F : float
        The frailty parameter.
    seed : int, optional
        Seed for the random number generator for reproducibility (default is 0).
    """

    np.random.seed(seed=seed)
    cases1 = stats.poisson(mu=lambda_ * F).rvs(z0)
    cases2 = stats.poisson(mu=lambda_ * F).rvs(z0)

    return cases1, cases2


def posterior_gamma(x, total_cases, T, lambda_, k, theta):
    """
    Calculates the posterior gamma distribution for the frailty parameter F
    given the observed data.
    Parameters:
    x : np.ndarray
        The values at which to evaluate the posterior distribution.
    total_cases : int
        Total number of cases observed.
    T : float
        Time period over which the cases are observed.
    lambda_ : float
        Rate parameter for the Poisson distribution of cases.
    k : float
        Shape parameter for the gamma distribution of the frailty.
    theta : float
        Scale parameter for the gamma distribution of the frailty.
    """
    return stats.gamma.pdf(
        x, a=k + total_cases, scale=1 / (1 / theta + 2 * lambda_ * T)
    )
