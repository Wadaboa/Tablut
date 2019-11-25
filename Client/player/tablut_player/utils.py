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
    for u in unwanted:
        try:
            seq.remove(u)
        except ValueError:
            pass
    return seq


def flatten_list(seq):
    flattened = []
    for sublist in seq:
        if isinstance(sublist, list):
            flattened.extend(flatten_list(sublist))
        else:
            flattened.append(sublist)
    return flattened


def get_from_set(seq):
    elem = seq.pop()
    seq.add(elem)
    return elem


def copy(obj):
    return cp.deepcopy(obj)


def get_rand(seq):
    if not isinstance(seq, list):
        seq = list(seq)
    random.seed(time.time())
    return random.choice(seq)


def get_rand_double(min_val, max_val):
    return random.uniform(min_val, max_val)


def get_rand_int(min_val, max_val):
    return random.randrange(min_val, max_val)
