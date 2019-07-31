from secml.array import CArray
from secml.figure import CFigure


# we must define a function that take an array and return a value for every row
def f(array):
    res = CArray.zeros(array.shape[0])
    for r in range(array.shape[0]):
        x = array[r, 0]
        y = array[r, 1]
        res[r] = x + y
    return res


fig = CFigure()

fig.sp.plot_fun(f, levels=[.5, 1, 1.5])

fig.sp.grid()
fig.show()
