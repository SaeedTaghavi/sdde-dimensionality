from init import *

darkcolors = ['#064877', '#A14C00', '#026502']

def Fdesc_inset(ax, left=.225, bottom=0.05, width=.4, height=.55):
    insetax = inset_axes(ax, width="100%", height="100%", loc='lower left',
                         bbox_to_anchor=(left, bottom, width, height), bbox_transform=ax.transAxes)
    insetax.set_facecolor('#EEEEEE')
    for spine in insetax.spines.values():
        spine.set_visible(False)
    insetax.set_title("$dx_0/dt = F(x_{\Delta}) + q ξ(t)$")
    α = -1; β = 1;
    xarr = np.linspace(-6*β, 6*β); yarr = TanhModel.F(xarr, α, β); insetax.plot(xarr, yarr);
    xarr = np.linspace(-β, β); yarr = LinearModel.F(xarr, α, β); insetax.plot(xarr, yarr);
    insetax.set_xlabel('$x$', labelpad=-4)
    insetax.set_ylim(2.2*α*β, -3.4*α*β)
    insetax.text(-6*β, -2.5*α*β, '$F(x) = 2αβ \\, \\tanh \\, x / 2β$');
    insetax.plot([-β, β], [-1.3*α*β, -1.3*α*β], color='#333333')
    insetax.annotate(s='$2β$', xy=(0, -1.4*α*β), xytext = (0, -1.5*α*β), horizontalalignment='center', color='#333333')
    insetax.plot([1.7*β, 1.7*β], [α*β, -α*β], color='#333333')
    insetax.annotate(s='$2αβ$', xy=(1.7*β, 0), xytext=(1.8*β, 0), verticalalignment='center', horizontalalignment='left', color='#333333')
    insetax.set_xticks([]);
    insetax.set_yticks([]);
