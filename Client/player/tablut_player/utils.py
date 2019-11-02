'''
Utility functions
'''


import tablut_player

__all__ = ['remove_unwanted_seq']


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
