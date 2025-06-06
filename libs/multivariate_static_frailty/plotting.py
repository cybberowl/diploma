import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from .math import bivariate_density_conditional_data_ndim
from .math import posterior_gamma_moments


def draw_matrix(ax, cov, mean):
    """Draw a matrix with the correlation and mean values."""

    assert cov.shape == (3, 3)

    ax.axvline(x=0.01, ymin=0.01, ymax=0.99, color="black")
    ax.axvline(x=0.33, ymin=0.01, ymax=0.99, color="black")
    ax.axvline(x=0.66, ymin=0.01, ymax=0.99, color="black")
    ax.axvline(x=0.99, ymin=0.01, ymax=0.99, color="black")

    ax.axhline(y=0.01, xmin=0.01, xmax=0.99, color="black")
    ax.axhline(y=0.33, xmin=0.01, xmax=0.99, color="black")
    ax.axhline(y=0.66, xmin=0.01, xmax=0.99, color="black")
    ax.axhline(y=0.99, xmin=0.01, xmax=0.99, color="black")

    ax.grid(False)
    x_ticks = [0.17, 0.5, 0.83]
    y_ticks = [0.17, 0.5, 0.83]
    ax.set_xticks(x_ticks, ["$F$", "$G_1$", "$G_2$"], fontsize="large")
    ax.set_yticks(y_ticks, ["$F$", "$G_1$", "$G_2$"], fontsize="large")

    for ix, x in enumerate(x_ticks):
        for iy, y in enumerate(y_ticks):
            if ix != iy:
                ax.text(
                    x,
                    y,
                    f"{cov[ix,iy]:.4f}",
                    ha="center",
                    va="center",
                    color="tab:red",
                    fontsize="medium",
                )

            if ix == iy:
                ax.text(
                    x,
                    y,
                    f"{cov[ix,iy]:.4f}",
                    ha="center",
                    va="center",
                    color="tab:green",
                    fontsize="medium",
                )
                ax.text(
                    x,
                    y - 0.05,
                    f"{mean[ix]:.4f}",
                    ha="center",
                    va="center",
                    color="tab:blue",
                    fontsize="medium",
                )

    red_patch = mpatches.Patch(color="tab:red", label="Correlation")
    blue_patch = mpatches.Patch(color="tab:blue", label="Mean")
    green_patch = mpatches.Patch(color="tab:green", label="Standard deviation")
    ax.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(1.1, 1.1))
    ax.set_aspect("equal")


def evolution_animation_ndim(
    y1,
    y2,
    z0,
    a1,
    b1,
    a2,
    b2,
    gamma,
    lambda_,
    kF,
    thetaF,
    kG_1,
    thetaG_1,
    kG_2,
    thetaG_2,
    cases1,
    cases2,
    w=0.5,
    name="gifs/posterior_density_ndim.gif",
    cmap="coolwarm",
):
    """
    Create an animation of the evolution of the posterior distribution
    of the bivariate density function and the posterior distribution of the frailty variables
    in a multivariate static frailty model.

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
    kF : float
        shape parameter of the gamma distribution for the first spouse
    thetaF : float
        scale parameter of the gamma distribution for the first spouse
    kG_1 : float
        shape parameter of the gamma distribution for the second spouse
    thetaG_1 : float
        scale parameter of the gamma distribution for the second spouse
    kG_2 : float
        shape parameter of the gamma distribution for the third spouse
    thetaG_2 : float
        scale parameter of the gamma distribution for the third spouse
    cases1 : array-like
        number of cases for the first spouse at each time point
    cases2 : array-like
        number of cases for the second spouse at each time point
    w : float, optional
        weight for the prior distribution, by default 0.5
    name : str, optional
        name of the output gif file, by default 'gifs/posterior_density_ndim.gif'
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
        density_data = bivariate_density_conditional_data_ndim(
            y1,
            y2,
            z0,
            a1,
            b1,
            a2,
            b2,
            gamma,
            kF,
            thetaF,
            kG_1,
            thetaG_1,
            kG_2,
            thetaG_2,
            total_cases1[T - 1],
            total_cases2[T - 1],
            lambda_,
            T,
            w,
        )

        grid_1 = np.sort(np.unique(y1))
        grid_2 = np.sort(np.unique(y2))
        CSF = ax_1.pcolormesh(grid_1, grid_2, density_data, cmap=cmap)

        plt.colorbar(CSF, ax=ax_1, cax=cax, orientation="horizontal")
        ax_1.set_title(
            r"$\widetilde f(\widetilde y_1,\widetilde y_2;z_0;N_{1,1},\dots,N_{2,T})$"
        )
        ax_1.set_xlabel("$\widetilde y_1$ (female)")
        ax_1.set_ylabel("$\widetilde y_2$ (male)")

        cov, mean = posterior_gamma_moments(
            total_cases1[T - 1],
            total_cases2[T - 1],
            T,
            lambda_,
            kF,
            thetaF,
            kG_1,
            thetaG_1,
            kG_2,
            thetaG_2,
            w,
        )

        draw_matrix(ax_2, cov, mean)
        ax_2.set_title("$g_{z_0}(F|N_{1,1},\dots,N_{2,T})$")
        cases_tmp1 = np.append(cases1[:T], np.zeros(z0 - T))
        cases_tmp2 = np.append(cases2[:T], np.zeros(z0 - T))
        ax_3.bar(np.arange(z0), cases_tmp1, width=0.3, label="female")
        ax_3.bar(np.arange(z0) + 0.3, cases_tmp2, width=0.3, label="male")
        ax_3.legend()
        ax_3.set_title(r"Total cases $N_{i,t}$")
        ax_3.set_xlabel("time $T$ elapsed since marriage")
        fig.suptitle(
            rf"Updating probabilities, $T$ = {T}, $\gamma$ = {gamma}, $z_0$ = {z0}, $k$ = {kF}, $\theta(z_0) = 1/k, \lambda$ = {lambda_}"
        )

    anim = animation.FuncAnimation(
        fig, animate, frames=range(1, z0 + 1), interval=200, repeat=True
    )
    writer = animation.PillowWriter(fps=2)
    anim.save(name, writer=writer)
