"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd
Copyright (C) 2019  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from distutils.core import setup
from setuptools import find_packages

setup(
    name='precision_fda_brain_biomarker',
    version='1.0.0',
    packages=find_packages(),
    url='roche.com',
    author='Gunther Jansen, Ying He, Elina Koletou, Rena Yang, Liming Li, '
           'Wenjin Li, Chenkai Lv, Paul Paczuski, Patrick Schwab',
    author_email='patrick.schwab@roche.com',
    license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.14.1",
        "tensorflow == 2.11.1",
        "Keras >= 1.2.2",
        "pandas >= 0.18.0",
        "h5py >= 2.6.0",
        "scikit-learn >= 0.19.0",
        "scikit-multilearn == 0.2.0",
        "xgboost == 0.90",
        "cxplain >= 1.0.2"
    ]
)
