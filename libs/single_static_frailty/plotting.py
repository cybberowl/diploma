import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import stats

from .math import bivariate_density_conditional_data, posterior_gamma


def evolution_animation(
    y1,
    y2,
    z0,
    a1,
    b1,
    a2,
    b2,
    gamma,
    lambda_,
    k,
    theta,
    F,
    cases1,
    cases2,
    density,
    density_F,
    name="gifs/posterior_density.gif",
    cmap="coolwarm",
):
    """Create an animation of the evolution of the posterior distribution
    of the bivariate density function and the posterior distribution of the frailty variable.
    The animation shows how the posterior distribution changes as more data is observed.
    Parameters
    ----------
    y1 : array-like
        grid of values for the age of the first spouse
    y2 : array-like
        grid of values for the age of the second spouse
    z0 : int
        age up to which spouses have survived
    a1 : float
        gompertz scale parameter for the first spouse
    b1 : float
        gompertz shape parameter for the first spouse
    a2 : float
        gompertz scale parameter for the second spouse
    b2 : float
        gompertz shape parameter for the second spouse
    gamma : float
        parameter from Freund's model
    lambda_ : float
        intensity of the Poisson process
    k : float
        shape parameter of the gamma distribution
    theta : float
        scale parameter of the gamma distribution
    F : float
        true value of the frailty variable
    cases1 : array-like
        number of cases for the first spouse at each time point
    cases2 : array-like
        number of cases for the second spouse at each time point
    density : array-like
        prior bivariate density function
    density_F : array-like
        bivariate density function with perfect knowledge of the frailty variable
    name : str, optional
        name of the output gif file, by default 'gifs/posterior_density.gif'
    cmap : str, optional
        colormap to use for the density plot, by default 'coolwarm'
    """

    total_cases1 = cases1.cumsum()
    total_cases2 = cases2.cumsum()

    fig = plt.figure(figsize=(10, 7), layout="constrained")
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=(1, 1),
        height_ratios=(3, 0.1, 1),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax_1 = fig.add_subplot(gs[0, 0])
    ax_1.set_aspect("equal")
    ax_2 = fig.add_subplot(gs[:2, 1])
    ax_2.set_aspect("equal")
    ax_3 = fig.add_subplot(gs[2, :])
    cax = fig.add_subplot(gs[1, 0])

    def animate(T):

        ax_1.clear()
        ax_2.clear()
        ax_3.clear()
        density_data = bivariate_density_conditional_data(
            y1,
            y2,
            z0,
            a1,
            b1,
            a2,
            b2,
            gamma,
            k,
            theta,
            total_cases1[T - 1] + total_cases2[T - 1],
            lambda_,
            T,
        )

        grid_1 = np.sort(np.unique(y1))
        grid_2 = np.sort(np.unique(y2))
        CSF = ax_1.pcolormesh(grid_1, grid_2, density_data, cmap=cmap)
        idx = np.unravel_index(density.argmax(), np.array(density).shape)
        p1 = ax_1.scatter(
            grid_1[idx[0]],
            grid_2[idx[1]],
            marker="x",
            color="red",
            label="prior max",
            s=100,
        )
        idx = np.unravel_index(density_F.argmax(), np.array(density_F).shape)
        p2 = ax_1.scatter(
            grid_1[idx[0]],
            grid_2[idx[1]],
            marker="x",
            color="blue",
            label="perfect knowledge max",
            s=100,
        )

        plt.colorbar(CSF, ax=ax_1, cax=cax, orientation="horizontal")
        ax_1.set_title(
            r"$\widetilde f(\widetilde y_1,\widetilde y_2;z_0;N_{1,1},\dots,N_{2,T})$"
        )
        ax_1.set_xlabel("$\widetilde y_1$ (female)")
        ax_1.set_ylabel("$\widetilde y_2$ (male)")
        ax_1.legend(loc="lower left")

        x = np.linspace(0, 3, 1000)
        gamma_post = posterior_gamma(
            x, total_cases1[T - 1] + total_cases2[T - 1], T, lambda_, k, theta
        )
        gamma_prior = stats.gamma.pdf(x, a=k, scale=theta)
        ax_2.plot(x, gamma_post, label="Posterior distribution", lw=4)
        ax_2.plot(x, gamma_prior, label="Prior distribution", lw=4)
        ax_2.vlines(
            x=F,
            ymin=0,
            ymax=np.max(gamma_post),
            ls="--",
            color="black",
            lw=2,
            label="True $F$",
        )
        ax_2.legend()
        ax_2.set_title("$g_{z_0}(F|N_{1,1},\dots,N_{2,T})$")
        cases_tmp1 = np.append(cases1[:T], np.zeros(z0 - T))
        cases_tmp2 = np.append(cases2[:T], np.zeros(z0 - T))
        ax_3.bar(np.arange(z0), cases_tmp1, width=0.3, label="female")
        ax_3.bar(np.arange(z0) + 0.3, cases_tmp2, width=0.3, label="male")
        ax_3.legend()
        ax_3.set_title(r"Total cases $N_{i,t}$")
        ax_3.set_xlabel("time $T$ elapsed since marriage")
        fig.suptitle(
            rf"Updating probabilities, $T$ = {T}, $\gamma$ = {gamma}, $z_0$ = {z0}, $k$ = {k}, $\theta(z_0) = 1/k, \lambda$ = {lambda_}"
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=range(1, z0 + 1), interval=200, repeat=True
    )
    writer = animation.PillowWriter(fps=2)
    anim.save(name, writer=writer)
