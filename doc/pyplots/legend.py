from secml.array import CArray
from secml.figure import CFigure

X = CArray.linspace(-3.14, 3.14, 256, endpoint=True)
C, S = X.cos(), X.sin()

fig = CFigure(fontsize=14)
fig.sp.plot(X, C, color='red', alpha=0.5, linewidth=1.0, linestyle='-', label="cosine")
fig.sp.plot(X, S, label="sine")

fig.sp.grid()
fig.sp.legend(loc=0)

fig.show()
