from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=14)

t = CArray.arange(0.01, 20.0, 0.01)
fig.sp.semilogx(t, (2 * 3.14 * t).sin())

fig.sp.grid()
fig.sp.title('semilogx')

fig.show()

