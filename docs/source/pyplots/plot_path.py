from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=14)
fig.sp.title("5-points path")

path = CArray([[2, 2], [3, 2], [4, 7], [5, 4], [1, 3]])

fig.sp.plot_path(path)

fig.sp.xlim(0, 6)
fig.sp.ylim(1, 8)

fig.show()

