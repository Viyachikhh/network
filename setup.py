from setuptools import setup


def install_requires(req_file):
    with open(req_file, 'r') as f:
        req = f.readlines()
        req = list(map(lambda x: x[:-1], req[:-1]))
    return req


setup(
    name='network',
    packages=['network'],
    install_requires=install_requires('requirements.txt'),
    url='https://github.com/Viyachikhh/network'
)
