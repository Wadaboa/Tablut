'''
Tablut player entry point
'''

import argparse

import tablut_player


def parse_args():
    '''
    Parse standard input arguments
    '''
    parser = argparse.ArgumentParser(description='Tablut client player')
    parser.add_argument(
        dest='role',
        choices={'White', 'Black'},
        help='tablut player role'
    )
    parser.add_argument(
        dest='timeout', action='store',
        help='given time to compute each move'
    )
    parser.add_argument(
        dest='server_sock', action='store',
        help='tablut server socket address'
    )
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='run the command in debug mode'
    )
    args = parser.parse_args()
    tablut_player.TIMEOUT = args.timeout
    tablut_player.SERVER_SOCK = args.server_sock
    tablut_player.DEBUG = args.debug
    role = args.role


def main():
    parse_args()
    pass
