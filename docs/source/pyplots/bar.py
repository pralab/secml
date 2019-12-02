from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=12)

n = 12
X = CArray.arange(n)
Y1 = (1 - X / float(n)) * (1.0 - 0.5) * CArray.rand((n,)) + 0.5
Y2 = (1 - X / float(n)) * (1.0 - 0.5) * CArray.rand((n,)) + 0.5

fig.sp.xticks([0.025, 0.025, 0.95, 0.95])
fig.sp.bar(X, Y1, facecolor='#9999ff', edgecolor='white')
fig.sp.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    fig.sp.text(x, y, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    fig.sp.text(x, -y - 0.02, '%.2f' % y, ha='center', va='top')

fig.sp.xlim(-.5, n-.5)
fig.sp.xticks(())
fig.sp.ylim(-1.25, 1.25)
fig.sp.yticks(())

fig.sp.grid()
fig.show()
