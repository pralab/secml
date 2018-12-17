"""
.. module:: DataUtils
   :synopsis: Collection of mixed utilities for own data structure 
 
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
 
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from secml.data import CDataset
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CRoc

__all__ = ['plot_roc_and_prob_density',
           'plot_distance_dens', 'visualize_data_distance',
           'visualize_kernel_distance','plot_prob_density']


def plot_roc_and_prob_density(ts_scores, ts):
    """
    plot roc and the probability density function of benign and malicious class 
    """
    xm, malicious_pdf = density_estimation(ts_scores[ts.Y == 1])
    xb, benign_pdf = density_estimation(ts_scores[ts.Y == 0])

    # compute roc 
    roc = CRoc()
    roc.compute(ts.Y, ts_scores)

    # plot roc and score probability density
    fig = CFigure(height=5, width=12)
    fig.subplot(1, 2, 1, sp_type='roc')
    fig.sp.plot_roc(roc.fpr, roc.tpr)
    fig.subplot(1, 2, 2)
    fig.sp.plot(xb, benign_pdf, label="ben pdf")
    fig.sp.plot(xm, malicious_pdf, label="mal pdf")
    fig.sp.legend()
    return fig

def plot_prob_density(ts_scores, ts):
    """
    plot probability density function of benign and malicious class 
    """
    xm, malicious_pdf = density_estimation(ts_scores[ts.Y == 1])
    xb, benign_pdf = density_estimation(ts_scores[ts.Y == 0])

    # plot roc and score probability density
    fig = CFigure(height=5, width=5)
    fig.sp.plot(xb, benign_pdf, label="ben pdf")
    fig.sp.plot(xm, malicious_pdf, label="mal pdf")
    fig.sp.legend()
    return fig

def plot_distance_dens(data, distance):
    """
    plot the probability density function of the distance between 
    benign class, malicious class and extra-class 
    """
    X, y = _sort_data(data)

    if distance == 'l2':
        sk_distance = euclidean_distances
    elif distance == 'l1':
        sk_distance = manhattan_distances
    else:
        sk_distance = manhattan_distances

    D = CArray(sk_distance(X.get_data(), X.get_data())).ravel()
    y += 1
    y = y.ravel().atleast_2d()
    Y_mat = y.T.dot(y).ravel()

    #         print "distance :"
    #         print D
    #         print "e^-gamma*d"
    #         print CArray.exp(self.classifier.kernel.gamma * D)

    x_ben, ben_pdf = density_estimation(D[Y_mat == 1])
    x_extra, extra_pdf = density_estimation(D[Y_mat == 2])
    x_mal, mal_pdf = density_estimation(D[Y_mat == 4])

    # plot roc and score probability density
    fig = CFigure(height=5, width=12)
    fig.sp.plot(x_ben, ben_pdf, label="ben")
    fig.sp.plot(x_extra, extra_pdf, label="extra")
    fig.sp.plot(x_mal, mal_pdf, label="mal")
    fig.sp.legend()
    fig.sp.title("distance distribution")
    return fig


def _sort_data(data):
    """
    return the pattern matrix with first all negative 
    and then all positive sample 
    """
    sorted_idx = data.Y.argsort()
    sorted_X = data.X[sorted_idx, :]
    sorted_Y = data.Y[sorted_idx]
    return sorted_X, sorted_Y


def visualize_data_distance(data, distance_type='l1'):
    """
    compute the distance matrix between samples 
    and visualized them with a color plot 
    """
    sorted_X, sorted_y = _sort_data(data)

    # plot training patterns distances
    if distance_type == 'l2':
        sk_distance = euclidean_distances
    elif distance_type == 'l1':
        sk_distance = manhattan_distances
    else:
        raise NotImplementedError("not recognized distance type!")

    D = CArray(sk_distance(sorted_X.get_data(), sorted_X.get_data()))
    distance_fig = CFigure(4, 4)
    distance_fig.sp.title("train patterns d")
    im = distance_fig.sp.imshow(D)
    distance_fig.sp.colorbar(im)
    return distance_fig


def visualize_kernel_distance(data, kernel):
    """
    compute the kernel matrix between samples 
    and visualized them with a color plot 
    """
    sorted_X, sorted_y = _sort_data(data)

    if kernel.class_type == 'laplacian-2g':
        K = kernel.k(sorted_X, sorted_X, sorted_y)
    else:
        K = kernel.k(sorted_X, sorted_X, sorted_y)
    distance_fig = CFigure(4, 4)
    im = distance_fig.sp.imshow(K)
    distance_fig.sp.colorbar(im)
    distance_fig.sp.title("train patterns K")
    return distance_fig
