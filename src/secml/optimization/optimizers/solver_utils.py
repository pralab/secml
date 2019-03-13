from secml.array import CArray
import itertools

__all__ = ['check_neighbours']

def check_neighbours(x, fun, constr, bounds, eta=1e-4):
    """
    Generate neighbours of the x point.
    Return true if x own the lower function value compared to his neighours.

    Returns:
    ----------
    ns: CArray
        array that contain x neighbours coordinate. 
    feas_ns: list 
        list of neighbours that belong into the feasible domain
    x_l_t_ns: boolean
        (stay for x lesser than neighbours). is True if x have a function value
        lower than all his neighbours that belong inside the constraint, false
        elsewhere.
    """
    # generate point neighbours
    n_features = x.size
# TODO: SO FAR WE ARE SUPPOSING THAT ETA IS EQUAL FOR ALL THE FEATURES
    # generate all the possible x modifications
    modif_lst = list(itertools.permutations([0 + eta, 0, 0 - eta], n_features))
    num_ns = len(modif_lst)

    ns = CArray.zeros((num_ns, n_features))
    for n_idx in xrange(num_ns):
        ns[n_idx, :] = x + modif_lst[n_idx]

    feas_ns_idx = CArray([]) 
    # find neighbours that belong into the feasible domain:
    for idx in xrange(ns.shape[0]):
        n = ns[idx, :] #i-th neighbour
        if not constr.is_violated(n) and not bounds.is_violated(n):
            feas_ns_idx = feas_ns_idx.append(idx)

    # check if x is the point with the lower function score between neighbours
    # that belongs into the feasible domain:

    # get the score for all the neighbours point into the feasible domain
    feas_ns = ns[feas_ns_idx]
    num_feas_ns = feas_ns.shape[0]  # number of neighbours into the feasible domain
    ns_scores = CArray[ fun(feas_ns[i, :] for i in xrange(num_feas_ns))] 
    x_l_t_ns = (x < ns_scores).all()

    return ns, feas_ns_idx, x_l_t_ns
