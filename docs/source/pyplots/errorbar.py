from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=16)
fig.title('Errorbars can go negative!')

fig.sp.xscale("symlog", nonposx='clip')
fig.sp.yscale("symlog", nonposy='clip')

x = CArray(10.0).pow(CArray.linspace(0.0, 2.0, 20))
y = x ** 2.0

fig.sp.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)

fig.sp.ylim(bottom=0.1)

fig.sp.grid()
fig.show()
