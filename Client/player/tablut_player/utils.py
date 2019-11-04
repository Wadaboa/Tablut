'''
Utility functions
'''


import tablut_player

__all__ = ['remove_unwanted_seq', 'flatten']


def remove_unwanted_seq(seq, unwanted):
    '''
    Remove every item in unwanted list from seq, if present
    '''
    for u in unwanted:
        try:
            seq.remove(u)
        except ValueError:
            pass
    return seq


def flatten(seq):
    flattened = []
    for sublist in seq:
        if isinstance(sublist, list):
            flattened.extend(flatten(sublist))
        else:
            flattened.append(sublist)
    return flattened
