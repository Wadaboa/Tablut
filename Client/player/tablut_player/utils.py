'''
Utility functions
'''


import copy as cp
import time
import random


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
