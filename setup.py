
from setuptools import setup, find_packages

setup(
    name='xinterp',
    description='xarray extension adding smart interpolation tools.',
    author='Sean McDevitt',
    author_email='mcdevitts@gmail.com',
    version='0.1',
    packages=find_packages(),
    install_requires=(
        'numpy',
        'scipy',
        'xarray',
    ),
)
