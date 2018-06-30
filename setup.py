from setuptools import setup

setup(
    name='sdde-dimensionality',
    version='0.1dev',
    packages=['sdde-dim'],
    install_requires=[
        'tqdm >= 4.23.4',
        'luigi >= 2.7.5',
        'sinn >= 0.1',       # Not publicly available yet
        'acr_tools >= 0.1',  # Not publicly available yet
        'theano_shim >= 0.2.3'
    ]
)
