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

__all__ = ['split_dataset', 'get_train_test_idx', 'density_estimation',
           'plot_roc_and_prob_density', 'density_estimation',
           'plot_distance_dens', 'visualize_data_distance',
           'visualize_kernel_distance','plot_prob_density']


def split_dataset(dataset, num_train=None, num_test=None):
    """
    Random train-test dataset split 

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    train_size : float, int, or None, optional  (default None) 
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the 
        absolute number of test samples. If None, the value is automatically
         set to the complement of the train size. If train size is also None,
        test size is set to 0.25.
    test_size : float, int, or None, optional (default None) 
        If float, should be between 0.0 and 1.0 and represent the proportion 
        of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.

    Notes
    -----
    train_size and test_size could not be both None. If one is
    set to None the other should be a float, representing a
    percentage, or an integer.
    """
    print type(dataset)
    # X, y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(dataset.X.get_data(),
                                                        dataset.Y.tondarray(),
                                                        train_size=num_train,
                                                        test_size=num_test)

    train_data = CDataset(CArray(X_train), CArray(y_train))
    test_data = CDataset(CArray(X_test), CArray(y_test))

    return train_data, test_data


def get_train_test_idx(dataset, num_train=None, num_test=None):
    """
    Random train-test dataset split 

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Parameters
    ----------
    train_size : float, int, or None, optional  (default None) 
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the 
        absolute number of test samples. If None, the value is automatically
         set to the complement of the train size. If train size is also None,
        test size is set to 0.25.
    test_size : float, int, or None, optional (default None) 
        If float, should be between 0.0 and 1.0 and represent the proportion 
        of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically
        set to the complement of the test size.

    Notes
    -----
    train_size and test_size could not be both None. If one is
    set to None the other should be a float, representing a
    percentage, or an integer.


    """

    # TODO: THIS IS NOT OPTIMIZED, IT IS JUST FOR MAKE IT WORK. CHANGE IT.
    X = CArray.arange(start=0, stop=dataset.num_samples, step=1)
    train_data_idx, test_data_idx, y_train, y_test = train_test_split(
        X.get_data(),
        dataset.Y.tondarray(), train_size=num_train, test_size=num_test)

    return train_data_idx.tolist(), test_data_idx.tolist()


def density_estimation(scores):
    kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(
        scores.atleast_2d().T.get_data())

    x = CArray.linspace(-5.0 + scores.min(), 5.0 + scores.max(), 200)
    x = x.atleast_2d().T

    pdf = CArray(kde.score_samples(x.get_data()))
    pdf = pdf.exp()
    return x, pdf


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
    fig.sp.plot_roc(roc.fp, roc.tp)
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
