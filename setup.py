from setuptools import find_packages, setup

setup(
    name='thz-deconvolution',
    packages=find_packages(),
    version='1.0.0',
    description='THz-TDS deconvolution library',
    author='Arnaud Demion',
    license='MIT',
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'zmq',
        'tqdm'
    ]
)
