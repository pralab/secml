from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=14)

t = CArray.arange(0.01, 20.0, 0.01)
fig.sp.semilogy(t, (-t / 5.0).exp())

fig.sp.title('semilogy')
fig.sp.grid()
fig.show()
