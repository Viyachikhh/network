from setuptools import setup

setup(
    name='network',
    version='0.1',
    packages=['network'],
    package_dir={'': 'network'},
    install_requires=[
        'numpy',
        'python-mnist'
    ],
    url='https://github.com/Viyachikhh/network'
)