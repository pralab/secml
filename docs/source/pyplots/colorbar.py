from secml.array import CArray
from secml.figure import CFigure


def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * (-x ** 2 - y ** 2).exp()

fig = CFigure(width=10, title="Colorbar Example")
fig.subplot(1, 2, 1)

x_linspace = CArray.linspace(-3, 3, 256)
y_linspace = CArray.linspace(-3, 3, 256)

X, Y = CArray.meshgrid((x_linspace, y_linspace))
c = fig.sp.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='hot')
fig.sp.colorbar(c)
fig.sp.title("Hot Contourf")
fig.sp.xticks(())
fig.sp.yticks(())

fig.subplot(1, 2, 2)
c = fig.sp.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='winter')
fig.sp.colorbar(c)
fig.sp.title("Cold Contourf")
fig.sp.xticks(())
fig.sp.yticks(())

fig.show()

