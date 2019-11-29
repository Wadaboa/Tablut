'''
Utility functions
'''


import copy as cp
import time
import timeit
import random
import threading
from functools import wraps


INF = float('inf')


def remove_unwanted_list(seq, unwanted):
    '''
    Remove every item in unwanted list from seq, if present
    '''
    for elem in unwanted:
        try:
            seq.remove(elem)
        except ValueError:
            pass
    return seq


def flatten_list(seq):
    '''
    Return a one-dimensional list from the given multi-dimensional list
    '''
    flattened = []
    for sublist in seq:
        if isinstance(sublist, list):
            flattened.extend(flatten_list(sublist))
        else:
            flattened.append(sublist)
    return flattened


def get_from_set(seq):
    '''
    Return an element from a set without removing it
    '''
    elem = seq.pop()
    seq.add(elem)
    return elem


def copy(obj):
    '''
    Return a deep copy of the given object
    '''
    return cp.deepcopy(obj)


def set_random_seed(seed=None):
    '''
    Set the random seed to the given one or to the current time
    '''
    seed = seed if seed is not None else time.time()
    random.seed(seed)


def get_rand(seq, num=1):
    '''
    Return the given number of random elements from the given sequence
    '''
    if isinstance(seq, set):
        return get_from_set(seq)
    seq = list(seq) if not isinstance(seq, list) else seq
    return random.choice(seq) if num == 1 else random.sample(seq, num)


def get_rand_double(min_val, max_val):
    '''
    Return a random double value in the range [min_val, max_val]
    or [min_val, max_val), extracted from a uniform distribution
    '''
    return random.uniform(min_val, max_val)


def probability(p):
    '''
    Return True with probability p
    '''
    return p > get_rand_double(0.0, 1.0)


def get_rand_int(min_val, max_val):
    '''
    Return a random integer value in the range [min_val, max_val]
    '''
    return random.randrange(min_val, max_val)


def random_perturbation(radius=0.001):
    '''
    Return a random number in the symmetric interval [-radius, radius]
    '''
    return random.uniform(-radius, radius)


def clip(value, lower=None, upper=None):
    '''
    Clip the given value in the interval [lower, upper] or (-inf, upper]
    or [lower, inf)
    '''
    return (
        lower if lower is not None and value < lower
        else upper if upper is not None and value > upper
        else value
    )


def run_once(func):
    '''
    A decorator that runs a function only once
    '''
    def wrapper(*args, **kwargs):
        try:
            return wrapper._once_result
        except AttributeError:
            wrapper._once_result = func(*args, **kwargs)
            return wrapper._once_result
    return wrapper


"""
def timing(doc):
    '''
    Decorator that computes the decorated function running time.
    '''
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> RT:
            start = timeit.default_timer()
            ret = func(*args, **kwargs)
            end = timeit.default_timer()
            elapsed = end - start
            # Add runtime to benchmarks, if function already present
            for bench in doc.benchmarks:
                if bench.func_name == func.__name__:
                    bench.add_runtime(elapsed)
                    return ret
            # Add function and runtime to benchmarks, if function not present
            bench = qiocr._qiocr.Benchmark(func.__name__)
            bench.add_runtime(elapsed)
            doc.benchmarks.append(bench)
            return ret
        return wrapper
    return decorate
"""


class InterruptableThread(threading.Thread):
    '''
    Thread that can be interrupted
    '''

    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self, daemon=True)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = None

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        '''
        Return the thread result
        '''
        return self._result


def timeout(seconds):
    '''
    Decorator that kills a function after the given timeout
    '''
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            thr = InterruptableThread(func, * args, **kwargs)
            thr.start()
            thr.join(seconds)
            if not thr.is_alive():
                return thr.result
            raise TimeoutError('Execution expired.')
        return wrapper
    return decorate
