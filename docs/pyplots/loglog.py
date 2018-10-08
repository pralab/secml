from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=14)
fig.title('loglog base 4 on x')

t = CArray.arange(0.01, 20.0, 0.01)
fig.sp.loglog(t, 20 * (-t / 10.0).exp(), basex=2)

fig.sp.grid()
fig.show()
