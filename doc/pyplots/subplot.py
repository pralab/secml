import numpy as np
import matplotlib.pyplot as plt
from secml.figure import CFigure

fig = CFigure(fontsize=16)

# create a new subplot
fig.subplot(2, 2, 1)
x = np.linspace(-np.pi, np.pi, 100)
y = 2*np.sin(x)
# function `plot` will be applied to the last subplot created
fig.sp.plot(x, y)

# subplot indices are are the same of the first subplot
# so the following function will be run inside the previous plot
fig.subplot(2, 2, 1)
y = x 
fig.sp.plot(x, y)

# create a new subplot
fig.subplot(2, 2, 3)
fig.sp.plot(x, y)

fig.subplot(2, 2, grid_slot=(1, slice(2)))
y = 2*np.sin(x)
fig.sp.plot(x, y)

plt.show()
