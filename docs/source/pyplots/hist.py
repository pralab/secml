from secml.array import CArray
from secml.figure import CFigure

fig = CFigure(fontsize=14)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * CArray.randn((10000,))
num_bins = 50
# the histogram of the data
n, bins, patches = fig.sp.hist(x, num_bins, density=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = bins.normpdf(mu, sigma)
fig.sp.plot(bins, y, 'r--')
fig.sp.xlabel('Smarts')
fig.sp.ylabel('Probability')
fig.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.subplots_adjust(left=0.15)

fig.sp.grid()
fig.show()
