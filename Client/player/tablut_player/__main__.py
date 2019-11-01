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
        '-t', '--timeout', dest='timeout', action='store',
        help='given time to compute each move', default=60
    )
    parser.add_argument(
        '-s', '--server-ip', dest='server_ip', action='store',
        help='tablut server ip address', default='127.0.0.1'
    )
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='run the command in debug mode'
    )
    args = parser.parse_args()
    tablut_player.TIMEOUT = args.timeout
    tablut_player.SERVER_IP = args.server_ip
    tablut_player.DEBUG = args.debug
    role = args.role
    tablut_player.player.test()


def main():
    parse_args()
    pass
