from secml.array import CArray
from secml.figure import CFigure

X = CArray.linspace(-3.14, 3.14, 256, endpoint=True)
C, S = X.cos(), X.sin()

fig = CFigure(fontsize=14)

fig.sp.plot(X, C, color='red', alpha=0.5, linewidth=1.0, linestyle='-')
fig.sp.plot(X, S)

fig.sp.xlabel("x", color='r', fontsize=10)

fig.show()
