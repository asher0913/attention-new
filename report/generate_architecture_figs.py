import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle

os.makedirs('artifacts', exist_ok=True)


def box(ax, xy, text, color='#1f77b4'):
    x, y = xy
    bb = FancyBboxPatch((x, y), 2.6, 1.0, boxstyle='round,pad=0.15',
                        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.18)
    ax.add_patch(bb)
    ax.text(x + 1.3, y + 0.5, text, ha='center', va='center', fontsize=9, color='#1f1f1f')


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=ArrowStyle('-|>', head_length=8, head_width=4),
                                lw=1.0, color='#444'))
    if label:
        ax.text((x1+x2)/2, y1+0.25, label, ha='center', va='center', fontsize=8, color='#333')


def fig_baseline():
    fig, ax = plt.subplots(figsize=(10, 3))
    box(ax, (0.2, 1.2), 'Input $x$')
    box(ax, (3.2, 1.2), 'Encoder $F_e$\\(VGG11-BN-SGM)')
    box(ax, (6.2, 1.2), 'Noise\\$\tilde z = z+\varepsilon$')
    box(ax, (9.2, 1.2), 'Decoder $F_d$\\Task loss $L_D$')
    box(ax, (6.2, -0.2), 'GMM surrogate\\$L_C^{\rm GMM}$')
    arrow(ax, 2.8, 1.7, 3.2, 1.7)
    arrow(ax, 5.8, 1.7, 6.2, 1.7, label='$z$')
    arrow(ax, 8.8, 1.7, 9.2, 1.7, label='$\tilde z$')
    arrow(ax, 6.7, 1.0, 6.7, 0.8)
    ax.text(7.2, 0.55, '$L_C$ backprop to $F_e$', fontsize=8)
    ax.text(10.0, 1.9, 'Attacker $A_\phi$\\(MIA recon)', ha='left', va='center', fontsize=8, color='#555')
    arrow(ax, 10.8, 1.4, 11.5, 1.4)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.6, 2.4)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('artifacts/fig_architecture_baseline.png', dpi=300)
    plt.close(fig)


def fig_surrogates():
    fig, ax = plt.subplots(figsize=(10, 4))
    # shared trunk
    box(ax, (0.2, 2.0), 'Input $x$')
    box(ax, (3.2, 2.0), 'Encoder $F_e$')
    box(ax, (6.2, 2.0), 'Noise\\$\tilde z$')
    box(ax, (9.2, 2.0), 'Decoder $F_d$\\$L_D$')
    arrow(ax, 2.8, 2.5, 3.2, 2.5)
    arrow(ax, 5.8, 2.5, 6.2, 2.5)
    arrow(ax, 8.8, 2.5, 9.2, 2.5)

    # branch gated
    box(ax, (5.0, 0.8), 'Gated moments\\$w,\mu_c,v_c$')
    box(ax, (7.6, 0.8), 'Hinge on log-var\\$L_C^{\rm gated}$')
    arrow(ax, 6.8, 2.0, 6.8, 1.5, label='class groups')
    arrow(ax, 6.6, 1.3, 7.6, 1.3)
    arrow(ax, 8.5, 1.1, 8.5, 1.9)
    ax.text(8.5, 1.45, '$\nabla L_C$', ha='left', va='center', fontsize=8)

    # branch slots
    box(ax, (5.0, -0.6), 'Slots $s^{(t)}$\\(iterative attn)')
    box(ax, (7.6, -0.6), 'Gated x-attn\\$L_C^{\rm slot}$')
    arrow(ax, 6.2, 2.0, 6.2, 0.2, label='tokens')
    arrow(ax, 6.8, -0.15, 7.6, -0.15)
    arrow(ax, 8.5, -0.4, 8.5, 1.9)
    ax.text(8.6, 0.5, '$\nabla L_C$', ha='left', va='center', fontsize=8)

    ax.text(0.2, 3.1, 'Two surrogates share trunk; only $L_C$ branch changes', fontsize=9, color='#333')
    ax.set_xlim(-0.2, 11.8)
    ax.set_ylim(-1.2, 3.4)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('artifacts/fig_architecture_surrogates.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    fig_baseline()
    fig_surrogates()
    print('Saved to artifacts/fig_architecture_baseline.png and fig_architecture_surrogates.png')
