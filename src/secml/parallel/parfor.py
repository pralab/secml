from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed


def parfor(task, processes, args):
    """Parallel For.

    Applies a function *task* to each argument in *args*,
    using a pool of concurrent processes.
    
    Parameters
    ----------
    task : function
        Function object that should process each element in args.
    processes : int
        Maximum number of concurrent processes to be used in the pool.
        If higher than multiprocessing.cpu_count(),
        all processor's cores will be used.
    args : any
        Iterable object, where each element is an argument for task.
    
    Returns
    -------
    out : iterable
        Iterable object containing the output of
        task(arg) for each arg in args.

    """
    # Don't try to spawn more processes than available CPUs
    num_cores = min(cpu_count(), processes)

    pool = Pool(processes=num_cores)
    return pool.map(task, args)


def parfor2(task, n_reps, processes, *args):
    """Parallel For.

    Run function `task` using each argument in `args` as input,
    using a pool of concurrent processes.
    The `task` should take as first input the index of parfor iteration.

    Parameters
    ----------
    task : function
        Function object that should process each element in `args`.
    n_reps : int
        Number of times the `task` should be run.
    processes : int
        Maximum number of concurrent processes to be used in the pool.
        If higher than `multiprocessing.cpu_count()`,
        all processor's cores will be used.
    args : any, optional
        Tuple with input arguments for `task`.

    Returns
    -------
    out : list
        List with iteration output, sorted (rep1, rep2, ..., repN).

    """
    # Don't try to spawn more processes than available CPUs
    num_cores = min(cpu_count(), processes)

    return Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(task)(i, *args) for i in range(n_reps))


if __name__ == "__main__":

    from math import factorial
    arguments = range(10)
    res = [factorial(z) for z in arguments]
    parres = parfor(factorial, 2, arguments)
    print(parres)

    def element_wise_power(idx, list_of_scalars):
        print("Repetition {:} started...".format(idx))
        list_of_scalars_pow = []
        for obj_idx, obj in enumerate(list_of_scalars):
            list_of_scalars_pow.append(list_of_scalars[obj_idx]**idx)
        print("Repetition {:} ended...".format(idx))
        return list_of_scalars_pow

    parout = parfor2(element_wise_power, 4, 2, ([j for j in range(10)]))
    print(parout)
