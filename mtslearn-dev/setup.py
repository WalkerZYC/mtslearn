from setuptools import setup, find_packages

setup(
    name='mtslearn',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python package for medical time-series data processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/mtslearn',
    packages=find_packages(include=['mtslearn', 'mtslearn.*']),
    install_requires=[
        'numpy',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
