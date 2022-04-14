
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from typing import Dict, List, Tuple

dpi = 600


def plot_mcf_vs_b_offset(
        x1: np.ndarray, x2: np.ndarray,
        y1: np.ndarray, y2: np.ndarray,
        label1: str, label2: str, out_file=None
) -> None:
    """
        Ploting MCF slope vs offset
    """
    plot_color = ["red", "blue"]
    plot_label = ["Parallel cloud", "Perpendicular cloud"]
    plot_marker = ['o', 'o']

    l, w = .1, .65
    b, h = l, w
    bh = lh = l + w + 0.02

    rec_scat = [l, b, w, h]
    rec_hisx = [l, bh, w, 0.2]
    rec_hisy = [lh, b, .2, h]

    # xy = np.vstack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    xy1 = np.vstack([x1, y1])
    z1 = st.gaussian_kde(xy1)(xy1)

    xy2 = np.vstack([x2, y2])
    z2 = st.gaussian_kde(xy2)(xy2)

    buff = 0.01
    lc = 0.4
    hc = 0.012
    rec_cax1 = [l + w/3, l + h - hc - buff, lc, hc]
    rec_cax2 = [l + w/3, l + h - hc - 3*buff - hc, lc, hc]

    plt.figure(1)
    # axes of different plot
    ax_sac = plt.axes(rec_scat)
    ax_hx = plt.axes(rec_hisx)
    ax_hy = plt.axes(rec_hisy)
    ax_c1 = plt.axes(rec_cax1)
    ax_c2 = plt.axes(rec_cax2)

    # Create own cmap
    # Scatter plot
    # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

    s1 = ax_sac.scatter(x1, y1, s=0.7, c=z1, marker='o',
                        alpha=1, cmap='hot', linewidths=0)
    s1.set_clim(0, 30000)
    cb1 = plt.colorbar(s1, cax=ax_c1, orientation='horizontal')
    cb1.set_ticks([0, 2000, 30000])
    cb1.set_ticklabels(['', '', r'$3\times 10^{4}$'])
    cb1.ax.tick_params(axis='x', direction='out', pad=-2, labelsize=6)
    ax_c1.minorticks_off()
    ax_c1.annotate('', xy=(2000/30000-0.01, -.6), xycoords='axes fraction', xytext=(1.01, -1.7),
                   arrowprops=dict(arrowstyle="-", color='black', linestyle='dotted', linewidth=0.5, alpha=1))
    ax_c1.annotate('', xy=(0, 0), xycoords='axes fraction', xytext=(0, -2),
                   arrowprops=dict(arrowstyle="-", color='black', linestyle='dotted', linewidth=0.5, alpha=1))
    # cb1.set_label('Parallel Cloud', labelpad=-12)

    s2 = ax_sac.scatter(x2, y2, s=0.7, c=z2, marker='o',
                        alpha=1, cmap='cool', linewidths=0)
    s2.set_clim(0, 2000)
    cb2 = plt.colorbar(s2, cax=ax_c2, orientation='horizontal')
    cb2.set_ticks([0, 2000])
    cb2.set_ticklabels([r'0', r'$2\times 10^3$'])
    cb2.ax.tick_params(axis='x', direction='in', pad=1, labelsize=6)
    ax_c2.minorticks_off()
    # cb2.ax.tick_params(labelcolor = 'white')
    # cb2.set_label('Perpendicular Cloud', labelpad=-12)

    ax_sac.text(0.05, 0.00478, 'Parallel cloud')
    ax_sac.text(0.05, 0.00455, 'Perpendicular cloud')

    ax_sac.set_xlabel("Cloud-field Offset [degree]")
    ax_sac.set_ylabel("Normalized MCF Slope [column density$^{-1}$]")

    ax_sac.set_ylim(0, 0.005)
    ax_sac.set_xlim(-0.01, np.pi/2+0.01)

    # legend_elements = [Line2D([0], [0], marker = plot_marker[0], color = 'red', label = plot_label[0], markersize = 5, lw = 0),
    #                    Line2D([0], [0], marker = plot_marker[1], color = 'blue', label = plot_label[1], markersize = 5, lw = 0)]

    # # ax_sac.set_facecolor((0.7, 0.7, 0.7))
    # ax_sac.legend(handles = legend_elements, loc = 'lower center', ncol = 2)
    ax_sac.set_xticks(np.linspace(0, np.pi/2, 10),
                      ["%d" % i for i in np.linspace(0, 90, 10)])
    ax_sac.set_yticks(np.linspace(0, 0.005, 6), [
                      "%.3f" % i for i in np.linspace(0, 0.005, 6)])
    # ax_sac.set_title('MCF Slope versus Cloud-field Offset')

    # histogram for offset
    bin1 = np.linspace(0, np.pi/2, 150)
    ax_hx.hist(x1, bins=bin1, density=True,
               color=plot_color[0], alpha=0.5, label='Parallel cloud')
    ax_hx.hist(x2, bins=bin1, density=True,
               color=plot_color[1], alpha=0.5, hatch='///////', histtype='step', lw=0, label='Perpendicular cloud')
    ax_hx.set_xlim(-0.01, np.pi/2+0.01)
    ax_hx.set_xticks(np.linspace(0, np.pi/2, 10),
                     ["" for i in np.linspace(0, 90, 10)])
    ax_hx.set_yticks(np.linspace(0, 20, 2), [
                     "" for i in np.linspace(0, 20, 2)])
    ax_hx.legend(loc='upper center', ncol=2)

    # histogram for offset
    bin2 = np.linspace(0, 0.005, 150)
    ax_hy.hist(y1, bins=bin2, density=True,
               color=plot_color[0], alpha=0.5, orientation='horizontal', label='Parallel cloud')
    ax_hy.hist(y2, bins=bin2, density=True, edgecolor=plot_color[1], alpha=0.5,
               orientation='horizontal', hatch='///////', histtype='step', lw=0, label='Perpendicular cloud')
    ax_hy.set_ylim(0, 0.005)
    ax_hy.set_yticks(np.linspace(0, 0.005, 6), [
                     "" for i in np.linspace(0, 0.005, 6)])
    ax_hy.set_xticks(np.linspace(0, 2000, 2), [
                     "" for i in np.linspace(0, 2000, 2)])

    # plt.savefig('./fig/mcf_vs_offset_%s.png' % fn, dpi = dpi)
    plt.show()

# TODO For statistic test. #######################
def plot_vertical_distribution_discrete(
        ax: plt.Axes,
        x_bottom: float, x_vertical: np.ndarray, den: np.ndarray,
        height: float = 0.5, dir: str = 'r', color: str = 'red', width: float = 0.8) -> None:
    """
        Plot veritical distribution.

        :param ax:          axs
        :oaram x_bottom:    bar bottom location
        :param x_vertical:  bar location
        :param den:         value of the distribution.
        :param height:      maximum height of the distribution, default is 0.5
        :param dir:         direction for the distribution, 'l' or 'r'
        :param color:       matplotlib color string or RGBA value.
        :param width:       width of each horizontal bar, default is 0.8
    """

    # create a copy to prevent modification of the original data.
    den = den.copy()
    x_vertical = x_vertical.copy()

    # scale the distribution.
    den = den*height

    if dir == 'r':
        for a, b in zip(den, x_vertical):
            ax.add_patch(Rectangle((x_bottom+a, b-width/2), -a,
                         width, facecolor=color, alpha=0.3))
    else:
        for a, b in zip(den, x_vertical):
            ax.add_patch(Rectangle((x_bottom-a, b-width/2), a, width,
                         facecolor=color, alpha=0.3))
     
def plot_vertical_distribution_continues(ax, x, y, den, dx, dir='r', color='red', w=0.8, alpha=0.5, fill=True, hatch=None):

    den = den.copy()
    y = y.copy()

    den = den*dx

    if dir == 'r':
        den = x - den
    else:
        den = x + den

    xy = np.stack([den, y], -1)

    ax.add_patch(Polygon(xy, facecolor=color,
                 alpha=alpha, fill=fill, hatch=hatch))

def plot_vertical_distribution_continues(
        ax: plt.Axes,
        x_bottom: float, x_vertical: np.ndarray, den: np.ndarray,
        height: float = 0.5, dir: str = 'r', color: str = 'red') -> None:
    """
        Plot veritical distribution.

        :param ax:          axs
        :oaram x_bottom:    bar bottom location
        :param x_vertical:  bar location
        :param den:         value of the distribution.
        :param height:      maximum height of the distribution, default is 0.5
        :param dir:         direction for the distribution, 'l' or 'r'
        :param color:       matplotlib color string or RGBA value.
    """

    # create a copy to prevent modification of the original data.
    den = den.copy()
    x_vertical = x_vertical.copy()

    # scale the distribution.
    den = den*height

    if dir == 'r':
        den = x_bottom - den
    else:
        den = x_bottom + den
        
    xy = np.stack([den, x_vertical], -1)
    
    ax.add_patch(Polygon(xy, facecolor=color,
                 alpha=0.5, fill=True, hatch=None))
    
       

def plot_statistic_test_discrete(
        x1: np.ndarray, y1: np.ndarray, li1: np.ndarray, bar_coord1: np.ndarray, prob1: np.ndarray, obs1: float,
        x2: np.ndarray, y2: np.ndarray, li2: np.ndarray, bar_coord2: np.ndarray, prob2: np.ndarray, obs2: float,
        xlim: Tuple, ylim: Tuple, bottom_scale: float=0.4, out_file: str = None,
        title: str = None, xlabel: str = None, ylabel: str = None, ylabel_likeli:str = None,
        likeli_label1: str = None, likeli_label2: str = None,
        obs_label1: str = None, obs_label2: str = None) -> None:
    """
        plot statistical test result.


        :param x1:          x value of the first data set.
        :param y1:          y value of the first data set.
        :param li1:         relative likelihood of the first data set.
        :param bar_coord1:  x-value of the probability
        :param prob1:       probability value of the first data set.
        :param x2:          x value of the second data set.
        :param y2:          y value of the second data set.
        :param li2:         relative likelihood of the second data set.
        :param bar_coord2:  x-value of the probability
        :param prob2:       probability value of the second data set.
        :param xlim:        xlimit for the main plot
        :param ylim:        ylimit for the main plot
        :param ylim2:       ylimit for the relative likelihood
        :param out_file:    full/relative path to the output file.
    """

    fig, ax1 = plt.subplots()

    # TODO Major plot  #################################################
    # Scatter plot with error bar
    ax1.scatter(x1, y1, color='red', marker='o', alpha=0.7, s=5)
    ax1.scatter(x2, y2, color='blue', marker='o', alpha=0.7, s=5)

    # Horizontal line for observation.
    if obs_label1:
        ax1.hlines(obs1, -1, 14,  color='red', ls='solid',
                   label=obs_label1)
    else:
        ax1.hlines(obs1, -1, 14,  color='red', ls='solid')

    if obs_label2:
        ax1.hlines(obs2, -1, 14,  color='blue', ls='dashed',
                   label=obs_label2)
    else:
        ax1.hlines(obs2, -1, 14,  color='blue', ls='dashed')

    # legend for the main plot.
    if obs_label1 or obs_label2:
        ax1.legend(loc='upper left')
    ####################################################################

    # TODO Vertical distribution (Normalize first) #####################
    prob1 = prob1.copy()
    prob2 = prob2.copy()
    prob1 /= np.max([np.max(prob1), np.max(prob2)])
    prob2 /= np.max([np.max(prob1), np.max(prob2)])

    # vertical distribution one by one
    for i, prob in zip(bar_coord1, prob1):
        plot_vertical_distribution_discrete(
            ax=ax1, x_bottom=i, x_vertical=bar_coord1, den=prob,
            height=0.9, dir='l', color='red', width=0.8)

    for i, prob in zip(bar_coord2, prob2):
        plot_vertical_distribution_discrete(
            ax=ax1, x_bottom=i, x_vertical=bar_coord2, den=prob,
            height=0.9, dir='r', color='blue', width=0.8)
    ####################################################################

    # TODO Set title and axis labels ###################################
    if title:
        ax1.set_title(title)
    if ylabel:
        ax1.set_ylabel(ylabel)
    if xlabel:
        ax1.set_xlabel(xlabel)
    ax1.set_ylim(*ylim)
    ax1.set_xlim(*xlim)

    ax1.set_xticks(np.arange(0, 14), ["%d" %
                   i for i in np.arange(0, 14)])
    ax1.set_yticks(np.arange(0, 14), ["%d" %
                   i for i in np.arange(0, 14)])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    ax1.yaxis.set_minor_locator(plt.NullLocator())
    #####################################################################

    # TODO small plot for relative likelihood ###########################
    ax2 = ax1.twinx()

    # bar plot for likelihood function
    w = 0.6  # default bar width.
    if likeli_label1:
        ax2.bar(x1+w/4, bottom_scale*li1, label=likeli_label1,
                width=w/2, color='red', alpha=0.5)
    else:
        ax2.bar(x1+w/4, bottom_scale*li1,
                width=w/2, color='red', alpha=0.5)

    if likeli_label2:
        ax2.bar(x2-w/4, bottom_scale*li2, label=likeli_label2, width=w/2,
                edgecolor='blue', alpha=0.5, fill=False, hatch='////', )
    else:
        ax2.bar(x2-w/4, bottom_scale*li2, width=w/2,
                edgecolor='blue', alpha=0.5, fill=False, hatch='////', )
    
    if ylabel_likeli:
        ax2.set_ylabel(ylabel_likeli, labelpad=10, loc='bottom')
    # Add legend if any labels.
    if likeli_label1 or likeli_label2:
        ax2.legend(loc='lower right', ncol=1)
    # Styling
    ax2.set_ylim(0, 1)
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.yaxis.set_minor_locator(plt.NullLocator())
    ax2.tick_params(axis='y', colors=(0.2, 0.2, 0.2))
    
    # Set y ticks for the likelihood function.
    ax2.set_yticks(np.linspace(0, bottom_scale, 5), labels=[
                   '%.2f' % i for i in np.linspace(0, 1, 5)])
    ####################################################################

    if out_file:
        plt.savefig(out_file, dpi=dpi)
    plt.show()

def plot_statistic_test_continues(
        x1: np.ndarray, y1: np.ndarray, li1: np.ndarray, bar_coord1: np.ndarray, prob1: np.ndarray, obs1: float,
        x2: np.ndarray, y2: np.ndarray, li2: np.ndarray, bar_coord2: np.ndarray, prob2: np.ndarray, obs2: float,
        xlim: Tuple, ylim: Tuple,
        vert_height: float = 0.6,
        obs1_std:float= None, obs2_std:float = None,
        bottom_scale: float=0.4, out_file: str = None,
        title: str = None, xlabel: str = None, ylabel: str = None, ylabel_likeli:str = None,
        likeli_label1: str = None, likeli_label2: str = None,
        obs_label1: str = None, obs_label2: str = None,) -> None:
    """
        plot statistical test result.


        :param x1:          x value of the first data set.
        :param y1:          y value of the first data set.
        :param li1:         relative likelihood of the first data set.
        :param bar_coord1:  x-value of the probability
        :param prob1:       probability value of the first data set.
        :param x2:          x value of the second data set.
        :param y2:          y value of the second data set.
        :param li2:         relative likelihood of the second data set.
        :param bar_coord2:  x-value of the probability
        :param prob2:       probability value of the second data set.
        :param xlim:        xlimit for the main plot
        :param ylim:        ylimit for the main plot
        :param ylim2:       ylimit for the relative likelihood
        :param out_file:    full/relative path to the output file.
    """

    fig, ax1 = plt.subplots()

    # TODO Major plot  #################################################
    # Scatter plot with error bar
    ax1.scatter(x1, y1, color='red', marker='o', alpha=0.7, s=5)
    ax1.scatter(x2, y2, color='blue', marker='o', alpha=0.7, s=5)

    # Horizontal line for observation.
    if obs_label1:
        ax1.hlines(obs1, -1, 14,  color='red', ls='solid',
                   label=obs_label1)
    else:
        ax1.hlines(obs1, -1, 14,  color='red', ls='solid')

    if obs_label2:
        ax1.hlines(obs2, -1, 14,  color='blue', ls='dashed',
                   label=obs_label2)
    else:
        ax1.hlines(obs2, -1, 14,  color='blue', ls='dashed')
    # sd of the observation.
    if obs1_std:
        ax1.vlines(13+0.5, obs1-obs1_std, obs1+obs1_std,  color='red', ls='solid')
        ax1.hlines(obs1-obs1_std, 13+0.4, 13+0.6, color='red', ls='solid')
        ax1.hlines(obs1+obs1_std, 13+0.4, 13+0.6, color='red', ls='solid')
    if obs2_std:
        ax1.vlines(13, obs2-obs2_std, obs2+obs2_std,  color='blue', ls='solid')
        ax1.hlines(obs2-obs2_std, 13-.1, 13+0.1, color='blue', ls='solid')
        ax1.hlines(obs2+obs2_std, 13-.1, 13+0.1, color='blue', ls='solid')
    # legend for the main plot.
    if obs_label1 or obs_label2:
        ax1.legend(loc='upper left')
    ####################################################################

    # TODO Vertical distribution (Normalize first) #####################
    prob1 = prob1.copy()
    prob2 = prob2.copy()
    prob1 /= np.max([np.max(prob1), np.max(prob2)])
    prob2 /= np.max([np.max(prob1), np.max(prob2)])

    # vertical distribution one by one
    for x_bot, prob in zip(x1, prob1):
        plot_vertical_distribution_continues(
            ax=ax1, x_bottom=x_bot, x_vertical=bar_coord1, den=prob,
            height=vert_height, dir='l', color='red')

    for x_bot, prob in zip(x2, prob2):
        plot_vertical_distribution_continues(
            ax=ax1, x_bottom=x_bot, x_vertical=bar_coord2, den=prob,
            height=vert_height, dir='r', color='blue')
    ####################################################################

    # TODO Set title and axis labels ###################################
    if title:
        ax1.set_title(title)
    if ylabel:
        ax1.set_ylabel(ylabel)
    if xlabel:
        ax1.set_xlabel(xlabel)
    ax1.set_ylim(*ylim)
    ax1.set_xlim(*xlim)

    ax1.set_xticks(np.arange(0, 14), ["%d" %
                   i for i in np.arange(0, 14)])
    # ax1.set_yticks(np.arange(0, 14), ["%d" %
    #                i for i in np.arange(0, 14)])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    # ax1.yaxis.set_minor_locator(plt.NullLocator())
    #####################################################################

    # TODO small plot for relative likelihood ###########################
    ax2 = ax1.twinx()

    # bar plot for likelihood function
    w = 0.6  # default bar width.
    if likeli_label1:
        ax2.bar(x1+w/4, bottom_scale*li1, label=likeli_label1,
                width=w/2, color='red', alpha=0.5)
    else:
        ax2.bar(x1+w/4, bottom_scale*li1,
                width=w/2, color='red', alpha=0.5)

    if likeli_label2:
        ax2.bar(x2-w/4, bottom_scale*li2, label=likeli_label2, width=w/2,
                edgecolor='blue', alpha=0.5, fill=False, hatch='////', )
    else:
        ax2.bar(x2-w/4, bottom_scale*li2, width=w/2,
                edgecolor='blue', alpha=0.5, fill=False, hatch='////', )
    
    if ylabel_likeli:
        ax2.set_ylabel(ylabel_likeli, labelpad=10, loc='bottom')
    # Add legend if any labels.
    if likeli_label1 or likeli_label2:
        ax2.legend(loc='upper right', ncol=1)
    # Styling
    ax2.set_ylim(0, 1)
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.yaxis.set_minor_locator(plt.NullLocator())
    ax2.tick_params(axis='y', colors=(0.2, 0.2, 0.2))
    
    # Set y ticks for the likelihood function.
    ax2.set_yticks(np.linspace(0, bottom_scale, 5), labels=[
                   '%.2f' % i for i in np.linspace(0, 1, 5)])
    ####################################################################

    if out_file:
        plt.savefig(out_file, dpi=dpi)
    plt.show()

