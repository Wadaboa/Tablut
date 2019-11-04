'''
Tablut package setup
'''

from setuptools import setup, find_packages

from tablut_player import __version__


setup(
    name='tablut_player',
    version=__version__,
    description='Fundamentals of Artificial Intelligence and Knowledge Representation assignement at UNIBO',
    author='Leonardo Calbi, Alessio Falai',
    author_email='leonardo.calbi@studio.unibo.it, alessio.falai@studio.unibo.it',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='==3.7.*',
    keywords='unibo ai tablut',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tablut_player = tablut_player.__main__:main'
        ]
    }
)
