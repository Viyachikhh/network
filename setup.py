from setuptools import setup

setup(
    name='network',
    version='0.1',
    packages=['dir_network.network'],
    package_dir={'': 'network'},
    install_requires=[
        'numpy'
    ],
    url='https://github.com/Viyachikhh/network'
)