from secml.data import CDataset
from secml.figure import CFigure

dataset = CDataset.create_random()

fig = CFigure(fontsize=14)
fig.sp.scatter(dataset.X[:, 0].ravel(),
               dataset.X[:, 1].ravel(),
               s=75, c=dataset.Y, alpha=.7)

fig.show()
