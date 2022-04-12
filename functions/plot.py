
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np


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
