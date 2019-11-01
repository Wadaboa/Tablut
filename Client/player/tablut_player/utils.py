'''
Utility functions
'''


import tablut_player


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
