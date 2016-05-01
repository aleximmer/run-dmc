# runDMC [![Build Status](https://travis-ci.com/AlexImmer/run-dmc.svg?token=RTEhNHKreGSnaC3U1jh2&branch=master)](https://travis-ci.com/AlexImmer/run-dmc)

## Dependencies
- [Git LFS](https://git-lfs.github.com/)
- Python 3 (tested with 3.5)
- Python.h for Python 3 (e.g. `sudo apt-get install python3.5-dev`)

## Installation
- Install Git Large File Storage **before** cloning the repository  
- run `git lfs install` in the cloned repo  

Optional
- Work in a virtual environment using virtualenv
- `sudo pip3 install virtualenv`  
- `virtualenv venv`
- `source venv/bin/activate`

Required
- Install requirements (NumPy, pandas, SciPy, ...)
- `pip3 install -r requirements.txt`

Confirm installation by running `python -m unittest discover` or `python3 -m unittest discover`

Modify IPython notebook files using e.g. `jupyter notebook features.ipynb`
