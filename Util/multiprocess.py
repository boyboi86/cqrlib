import sys
import time
import platform
import datetime as dt

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
import types

p = print

p_cpu = p('Num of CPU core: ', cpu_count())
p_platform = p('Machine info: ', platform.platform())
p_ver = p('Python', sys.version)
p_np_ver = p('Numpy', np.__version__)
p_pd_ver = p('Pandas', pd.__version__)

try: 
    import copy_reg
    p_cpu
    p_platform
    p_ver
    p_np_ver
    p_pd_ver
except: 
    import copyreg as copy_reg
    p_cpu
    p_platform
    p_ver #python 3 syntax diff from python 2
    p_np_ver
    p_pd_ver


def _pickle_method(method):
    func_name=method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try: 
            func=cls.__dict__[func_name]
        except KeyError:
            pass
        else: break
    return func.__get__(obj,cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def opt_num_threads(Idx: pd.DataFrame.index, num_threads: int, mp_batches: int):
    '''
    experimental
    only for idx_matrix if error try using num_threads = 1, mp_batches = 1
    basically this func is to reduce work load by breaking down certain work.
    
    it is dependent on the molecule total count, and splitting atom.
    
    params: rowIdx => return pandas.DataFrame rows
    params: num_Threads => minimal num of threads for multiprocessing
    '''
    opt_num_thread = [0, 0]
    for i in np.arange(24, 0, -1): # experimental
        if opt_num_thread[0] == 0 and Idx.shape[0] % i == 0:
            opt_num_thread[0] = i
        if opt_num_thread[1] == 0 and Idx.shape[0] % i == 1:
            opt_num_thread[1] = i
    _max_num_threads = max(num_threads, max(opt_num_thread[0], opt_num_thread[1]))
    _num_processes = max(mp_batches, min(opt_num_thread[0], opt_num_thread[1]))
    p(_max_num_threads, _num_processes)
    return _max_num_threads, _num_processes

# Snippet 20.5 (page 306), the lin_parts function
def lin_parts(num_atoms, num_threads):
    """
    Advances in Financial Machine Learning, Snippet 20.5, page 306.
    The lin_parts function
    The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
    where the number of subsets is the minimum between the number of processors and the number
    of atoms. For N subsets we need to find the N+1 indices that enclose the partitions.
    This logic is demonstrated in Snippet 20.5.
    This function partitions a list of atoms in subsets (molecules) of equal size.
    An atom is a set of indivisible set of tasks.
    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with a single loop
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int) #if you use astype int, some func will never run. but int can have sequences
    return parts


# Snippet 20.6 (page 308), The nested_parts function
def nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    Advances in Financial Machine Learning, Snippet 20.6, page 308.
    The nested_parts function
    This function enables parallelization of nested loops.
    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :param upper_triangle: (bool) Flag to order atoms as an upper triangular matrix (including the main diagonal)
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with an inner loop
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)

    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upper_triangle:  # The first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)

    return parts


# Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book
def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, axis=0, **kargs):
    """
    AFML, Snippet 20.7, page 310.
    The mpPandasObj, used at various points in the book
    Parallelize jobs, return a dataframe or series.
    Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)
    First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
    or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
    than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
    where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
    cores will be idle half of the runtime, waiting for the first core to process molecule 1.
    Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
    every core will receive equal workload, even though the first 10 molecules take as much time as the
    next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
    mpBatches=1.
    Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
    a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
    the molecule.
    Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
    otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
    debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
    Once the code is debugged, we will want to use numThreads>1.
    Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.
    :param func: (function) A callback function, which will be executed in parallel
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                    Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param mp_batches: (int) Number of parallel batches (jobs per core)
    :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
    :param kargs: (var args) Keyword arguments needed by func
    :return: (pd.DataFrame) of results
    
    Pls note, slight modification to this snippet for optimised result
    """

    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out

    for i in out: df0 = df0.append(i)
    df0 = df0.sort_index(axis=axis) #experimental
    return df0


# Snippet 20.8, pg 311, Single thread execution, for debugging
def process_jobs_(jobs, task = None):
    """
    Advances in Financial Machine Learning, Snippet 20.8, page 311.
    Single thread execution, for debugging
    Run jobs sequentially, for debugging
    :param jobs: (list) Jobs (molecules)
    :return: (list) Results of jobs
    """

    time0 = time.time()
        
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)
    _report_progress(time0)
    return out


# Snippet 20.10 Passing the job (molecule) to the callback function
def expand_call(kargs):
    """
    Advances in Financial Machine Learning, Snippet 20.10.
    Passing the job (molecule) to the callback function
    Expand the arguments of a callback function, kargs['func']
    :param kargs: Job (molecule)
    :return: Result of a job
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out

def _report_progress(time0):
    """
    Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.
    Example of Asynchronous call to pythons multiprocessing library
    :param job_num: (int) Number of current job
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    # Report progress as asynch jobs are completed
    t_msg = (time.time() - time0) / 60.0
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    
    msg = "{0} done after {1:.2f} mins".format(time_stamp, t_msg)

    sys.stderr.write(msg + '\n')

# Snippet 20.9.1, pg 312, Example of Asynchronous call to pythons multiprocessing library
def report_progress(job_num, num_jobs, time0, task):
    """
    Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.
    Example of Asynchronous call to pythons multiprocessing library
    :param job_num: (int) Number of current job
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    #msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
    #      str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    msg = "{0} {1}% {2} done after {3} mins. Remaining {4} mins.".format(time_stamp, 
                                                                           round(msg[0] * 100, 2),
                                                                           task,
                                                                           round(msg[1], 2),
                                                                           round(msg[2], 2))
    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')


# Snippet 20.9.2, pg 312, Example of Asynchronous call to pythons multiprocessing library
def process_jobs(jobs, task=None, num_threads: int = 24):
    """
    Advances in Financial Machine Learning, Snippet 20.9.2, page 312.
    Example of Asynchronous call to pythons multiprocessing library
    Run in parallel. jobs must contain a 'func' callback, for expand_call
    :param jobs: (list) Jobs (molecules)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :return: (None)
    """

    if task is None:
        task = jobs[0]['func'].__name__

    pool = Pool(processes=num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, start = 1):
        out.append(out_)
        print(out, "this out")
        report_progress(i, len(jobs), time0, task)

    pool.close()
    pool.join()  # This is needed to prevent memory leaks
    return out