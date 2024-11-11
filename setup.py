from setuptools import setup, find_packages

setup(
    name='mtslearn',
    version='0.0.1',
    author='Walker ZYC',
    author_email='zycwalker11@gmail.com',
    description='A Python Package for ML using Irregularly Sampled Medical Time Series Data',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/WalkerZYC/mtslearn',
    packages=find_packages(where='mtslearn-dev/mtslearn'),  # Ensure this matches the configuration in pyproject.toml
    install_requires=[
        'numpy>=1.21.2',
        'pandas>=1.5.3',
        'matplotlib>=3.6.0',
        'seaborn>=0.11.2',
        'scikit-learn>=1.0.2',
        'shap>=0.41.0',
        'xgboost>=1.5.0',
        'lifelines>=0.26.4',
        'imbalanced-learn>=0.9.0',
        'fancyimpute>=0.7.0',
    ],
    extras_require={
        'dev': ['pytest', 'flake8', 'black'],  # Additional tools for development
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.6',
    include_package_data=True,  # Ensure non-code files are included in the package
    zip_safe=False,             # Avoid issues when installing the package
)
