from secml.array import CArray
from secml.figure import CFigure
from secml.core.constants import pi

X = CArray.linspace(-3.14, 3.14, 256, endpoint=True)
C, S = X.cos(), X.sin()

fig = CFigure(fontsize=14)

fig.sp.plot(X, C, color='red', alpha=0.5, linewidth=1.0, linestyle='-')
fig.sp.plot(X, S)

fig.sp.xticks(CArray([-pi, -pi / 2, 0, pi / 2, pi]))
fig.sp.xticklabels(CArray(["- pi", "-pi/2", "0", "pi/2", "pi"]))
fig.sp.tick_params(direction='out', length=6, width=2, colors='r', right=False)
fig.sp.yticks(CArray([-1, 0, +1]))

fig.show()
