from secml.data.loader import CDLRandom
from secml.figure import CFigure

dataset = CDLRandom().load()

fig = CFigure(fontsize=14)
fig.sp.scatter(dataset.X[:, 0].ravel(),
               dataset.X[:, 1].ravel(),
               s=75, c=dataset.Y, alpha=.7)

fig.show()
