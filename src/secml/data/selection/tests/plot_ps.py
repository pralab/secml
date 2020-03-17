from secml.figure.c_figure import CFigure
from secml.data.selection import CPrototypesSelector
from secml.data.loader import CDLRandomBlobs

dataset = CDLRandomBlobs(n_features=2, n_samples=30,
                         centers=[[-0.5, 0], [0.5, 1]],
                         cluster_std=(0.8, 0.8), random_state=7545).load()

fig = CFigure(width=6, height=2, markersize=8, fontsize=11)

rules = ['center', 'border', 'spanning', 'k-medians']
for rule_id, rule in enumerate(rules):

    ps = CPrototypesSelector.create(rule)
    ps.verbose = 2
    ds_reduced = ps.select(dataset, n_prototypes=5)

    fig.subplot(1, len(rules), rule_id+1)

    # Plot dataset points
    fig.sp.scatter(dataset.X[:, 0], dataset.X[:, 1], linewidths=0, s=30)
    fig.sp.plot(ds_reduced.X[:, 0], ds_reduced.X[:, 1], linestyle='None',
                markeredgewidth=2, marker='o', mfc='red')
    fig.sp.title('{:}'.format(rule))

    fig.sp.yticks([])
    fig.sp.xticks([])

    fig.sp.grid(False)

fig.tight_layout()
fig.show()
