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


def set_random_seed():
    '''
    Set the random seed to the current time
    '''
    random.seed(time.time())


def get_rand(seq):
    '''
    Return a random element from the given sequence
    '''
    set_random_seed()
    seq = list(seq) if not isinstance(seq, list) else seq
    return random.choice(seq)


def get_rand_double(min_val, max_val):
    '''
    Return a random double value in the range [min_val, max_val]
    or [min_val, max_val), extracted from a uniform distribution
    '''
    return random.uniform(min_val, max_val)


def get_rand_int(min_val, max_val):
    '''
    Return a random integer value in the range [min_val, max_val]
    '''
    return random.randrange(min_val, max_val)
