# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from tablut_player import __version__

with open('README.md') as f:
    README = f.read()

with open('../LICENSE') as f:
    LICENSE = f.read()

setup(
    name='tablut_player',
    version=__version__,
    description='Fundamentals of Artificial Intelligence and Knowledge Representation assignement at UNIBO',
    long_description=README,
    license=LICENSE,
    author='Leonardo Calbi, Alessio Falai',
    author_email='leonardo.calbi@studio.unibo.it, alessio.falai@studio.unibo.it',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='==3.7.4',
    keywords='unibo ai tablut',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tablut_player = tablut_player.player:main'
        ]
    }
)
