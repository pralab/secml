from secml.array import CArray
from secml.figure import CFigure

n = 5
fig = CFigure()

x = CArray.arange(100)
y = 3. * CArray.sin(x * 2. * 3.14 / 100.)

for i in range(n):
    temp = 510 + i
    sp = fig.subplot(n, 1, i)
    fig.sp.plot(x, y)
    # for add space from the figure's border you must increased default value parameters
    fig.subplots_adjust(bottom=0.4, top=0.85, hspace=0.001)
    fig.sp.xticklabels(())
    fig.sp.yticklabels(())

fig.show()
