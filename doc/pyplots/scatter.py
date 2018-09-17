from prlib.data import CDataset
from prlib.figure import CFigure

dataset = CDataset.create_random()

fig = CFigure(fontsize=14)
fig.sp.scatter(dataset.X[:, 0], dataset.X[:, 1], s=75, c=dataset.Y, alpha=.7)

fig.show()
