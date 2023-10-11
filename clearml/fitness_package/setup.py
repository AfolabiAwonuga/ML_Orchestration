from setuptools import setup
from setuptools import find_packages
import version

setup(
    name = 'fitness_package',
    version = version.__version__,
    description = (
        'Fitness project clearml test.'
    ),
    author = 'Afolabi',
    author_email = '',
    packages = find_packages(),
    install_requires = [
        'xgboost', 
        'scikit-learn',
        'matplotlib', 
        'clearml-agent',
        'clearml',
        'clearml[s3]'
    ],
    classifiers=[
        'Intended Audience :: Internal Use ONLY',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
