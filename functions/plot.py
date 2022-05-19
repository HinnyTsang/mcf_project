
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from typing import Dict, List, Tuple

# Default styling
dpi = 600

plt.rcParams['figure.dpi'] = dpi
plt.rcParams['savefig.dpi'] = dpi
plt.style.use(['science','ieee', 'scatter'])
plt.rcParams['figure.figsize'] = 5,5

# TODO For projected parameters ##################
# See check_projection.ipynb
def plot_scatter_with_hist(
        x1: np.ndarray, x2: np.ndarray,
        y1: np.ndarray, y2: np.ndarray,
        label1: str, label2: str, 
        xlabel:str, ylabel: str,
        xlim: Tuple, ylim:Tuple,
        abbr1: str=None, abbr2: str=None, 
        out_file=None
) -> None:
    """
        Ploting MCF slope vs offset
    """
    plot_color = ["red", "blue"]
    plot_marker = ['o', 'o']

    # setting of the grids #########################################
    l, w = .1, .65
    b, h = l, w
    bh = lh = l + w + 0.0

    # plotting area
    # [left, bottom, width, height]
    rec_scat = [l, b, w, h]
    rec_hisx = [l, bh, w, 0.2]
    rec_hisy = [lh, b, .2, h]

    # color bar
    buff = 0.02
    lc = 0.4
    hc = 0.012
    sep = 0.02
    rec_cax1 = [l + w/3, l + h - hc - buff, lc, hc]
    rec_cax2 = [l + w/3, l + h - 2*hc - buff - sep, lc, hc]
    ################################################################

    # For density showing ##########################################
    xy1, xy2 = np.vstack([x1, y1]), np.vstack([x2, y2])
    z1, z2 = st.gaussian_kde(xy1)(xy1), st.gaussian_kde(xy2)(xy2)
    ################################################################


    # axes of different plot #######################################
    plt.figure(1)
    ax_sac = plt.axes(rec_scat)
    ax_hx = plt.axes(rec_hisx)
    ax_hy = plt.axes(rec_hisy)
    ax_c1 = plt.axes(rec_cax1)
    ax_c2 = plt.axes(rec_cax2)
    ################################################################


    # Color bar ####################################################
    clim1, clim2 = np.max(z1), np.max(z2) 
    ################################################################


    # Plot Scatter and color bar ###################################
    s1 = ax_sac.scatter(x1, y1, s=700/len(z1), c=z1, marker=plot_marker[0], alpha=1, cmap='hot', linewidths=0)
    s2 = ax_sac.scatter(x2, y2, s=700/len(z2), c=z2, marker=plot_marker[1], alpha=1, cmap='cool', linewidths=0)

    s1.set_clim(0, np.max(clim1))
    s2.set_clim(0, np.max(clim2))

    cb1 = plt.colorbar(s1, cax=ax_c1, orientation='horizontal')
    cb2 = plt.colorbar(s2, cax=ax_c2, orientation='horizontal')

    if clim2 > clim1:
        cb1.set_ticks([0, clim1])
        cb2.set_ticks([0, clim1, clim2])
        cb1.set_ticklabels(['', f'{clim1:.2e}'])
        cb2.set_ticklabels([r'0', f'{clim1:.2e}' , f'{clim2:.2e}'])

    # current case, only tested for this.
    elif clim2 < clim1:
        cb1.set_ticks([0, clim2, clim1])
        cb2.set_ticks([0, clim2])
        cb1.set_ticklabels([r'', '', f''])
        cb2.set_ticklabels(['0', f'{clim2/clim1:.2f}'])

    # connection between two color bar. 
    ax_c1.annotate('', xy=(clim2/clim1, -.7), xycoords='axes fraction', xytext=(1, -1.7),
                    arrowprops=dict(arrowstyle="-", color='black', linestyle='solid', linewidth=0.5, alpha=1))
    ax_c1.annotate('', xy=(0, 0), xycoords='axes fraction', xytext=(0, -3),
                    arrowprops=dict(arrowstyle="-", color='black', linestyle='solid', linewidth=0.5, alpha=1))

    # Tick setting for the color bar
    cb1.ax.tick_params(axis='x', direction='out', pad=-2, labelsize=6)
    cb2.ax.tick_params(axis='x', direction='in', pad=1, labelsize=6)
    ax_c1.minorticks_off()
    ax_c2.minorticks_off()

    # label for the color bar
    ax_sac.annotate(text=label1, xy = (hc+buff, (h - hc - buff)/h), xycoords='axes fraction')
    ax_sac.annotate(text=label2, xy = (hc+buff, (h - hc*2 - buff-sep)/h), xycoords='axes fraction')
    ################################################################


    # Scatter axis setting #########################################
    if xlabel:
        ax_sac.set_xlabel(xlabel)
    if ylabel:
        ax_sac.set_ylabel(ylabel)

    # set xlim for scatter and the histogram 
    ax_sac.set_ylim(*ylim)
    ax_sac.set_xlim(*xlim)
    ax_hy.set_ylim(*ylim)
    ax_hx.set_xlim(*xlim)
    ################################################################


    # histogram for offset #########################################
    bin1 = np.linspace(xlim[0], xlim[1], int(np.sqrt(len(x1))))
    ax_hx.hist(x1, bins=bin1, density=True,
                color=plot_color[0], alpha=0.5, label=label1)
    ax_hx.hist(x2, bins=bin1, density=True, edgecolor=plot_color[1], 
            alpha=0.5, histtype='step', hatch='//////', lw=0, label=label2)

    ax_hx.set_xticklabels([])
    ax_hx.set_yticks([])
    ax_hx.legend(loc='upper center', ncol=2)
    ################################################################

    # histogram for the parameter
    bin2 = np.linspace(ylim[0], ylim[1], int(np.sqrt(len(x1))))
    ax_hy.hist(y1, bins=bin2, density=True,
                color=plot_color[0], alpha=0.5, orientation='horizontal', label= abbr1 if abbr1 else label1)
    ax_hy.hist(y2, bins=bin2, density=True, edgecolor=plot_color[1], alpha=0.5,
                orientation='horizontal', histtype='step', hatch='//////', lw=0, label=abbr2 if abbr2 else label2)

    ax_hy.set_yticklabels([])
    ax_hy.set_xticks([])
    ax_hy.legend(loc='lower center', ncol=1)
    ################################################################

    if out_file:
        plt.savefig(out_file, dpi = dpi)
    plt.show()

# TODO For statistic test. #######################
# See get_stats_tets_result.py / chech_stat_test_result.ipynb
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
    prob1 /= np.nanmax([np.nanmax(prob1), np.nanmax(prob2)])
    prob2 /= np.nanmax([np.nanmax(prob1), np.nanmax(prob2)])

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

def plot_likelihood(
    x1:np.ndarray, y1:np.ndarray,
    x2:np.ndarray, y2:np.ndarray,
    label1:str = None, label2:str=None,
    title:str=None, xlabel:str=None, ylabel:str=None,
    out_file:str=None) -> None:
    """
    plot total likelihood.
    """
    
    
    fig, ax = plt.subplots()
    
    
    # width of each bar
    w = 0.4
    if label1:
        ax.bar(x1+w/2, y1 , label = label1, width = w, color = 'red', alpha = 0.5, fill = True,)
    else:
        ax.bar(x1+w/2, y1 , width = w, color = 'red', alpha = 0.5, fill = True,)
    
    if label2:
        ax.bar(x2-w/2, y2 , label = label2, width = w, edgecolor = 'blue', alpha = 0.5, fill = False, hatch = '////')
    else:
        ax.bar(x2-w/2, y2, width = w, edgecolor = 'blue', alpha = 0.5, fill = False, hatch = '////')


    # Text lable for each bar.
    for x, y, s in zip([x1, x2], [y1, y2], [1, -1]):

        l = ["%.2f" % (i) for i in y]

        for i in range(len(x)):
            ax.text(x[i]+s*w/2, y[i]+0.015, s = l[i], ha = 'center', rotation = 'vertical', fontsize='small')

    # set ticks
    ax.set_xticks(np.arange(0, 14), ["%d"%i for i in np.arange(0, 14)])
    ax.set_yticks(np.arange(0, 14)/10, ["%.2f"% (i/10) for i in np.arange(0, 14)])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(0, 1.07)
    
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # set legend
    if label1 or label2:
        plt.legend()
    # save output file        
    if out_file:
        plt.savefig(out_file, dpi = dpi)
    plt.show()