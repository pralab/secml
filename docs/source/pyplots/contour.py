from secml.array import CArray
from secml.figure import CFigure


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * (-x ** 2 - y ** 2).exp()

fig = CFigure()

x_linspace = CArray.linspace(-3, 3, 256)
y_linspace = CArray.linspace(-3, 3, 256)

X, Y = CArray.meshgrid((x_linspace, y_linspace))

C = fig.sp.contour(X, Y, f(X, Y), linewidths=.5, cmap='hot')

fig.sp.xticks(())
fig.sp.yticks(())

fig.show()

