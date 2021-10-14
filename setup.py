from os import listdir
from setuptools import setup, find_packages

setup(
    name='network',
    version='0.1',
    packages=list(map(lambda x: 'network.' + x, listdir('network/'))),
    install_requires=[
        'numpy'
    ],
    url='https://github.com/Viyachikhh/network'
)