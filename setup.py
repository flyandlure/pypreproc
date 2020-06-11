from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pypreproc',
    packages=['pypreproc'],
    version='0.11',
    license='MIT',
    description='PyPreProc is a Python package for correcting, converting, clustering and creating data in Pandas as part of the feature engineering step of the machine learning process.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matt Clarke',
    author_email='matt@flyandlure.org',
    url='https://github.com/flyandlure/pypreproc',
    download_url='https://github.com/flyandlure/pypreproc/archive/master.zip',
    keywords=['python', 'pandas', 'preprocessing', 'feature engineering', 'features', 'maths'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['pandas', 'lifetimes', 'numpy', 'sklearn']
)
