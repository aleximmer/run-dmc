# runDMC [![Build Status](https://travis-ci.com/AlexImmer/run-dmc.svg?token=RTEhNHKreGSnaC3U1jh2&branch=master)](https://travis-ci.com/AlexImmer/run-dmc)

##
- Instead we now use onedrive as data source, so you have to download and place files yourself. This also makes feature merging obsolete as the first two groups already do that for us.
- **Download** 2016-04-30-07-data.gz from [onedrive](https://onedrive.live.com/?authkey=%21AAjJc4NIZ1ot97U&id=876D0040AD5E0EBE%213618&cid=876D0040AD5E0EBE)
- Put both files to *data/*
- After the first call of *processed_data()* the cleansed and preprocessed file will be cached (processing will take from 20 to 120 minutes depending on features selected)

## Dependencies
- Python 3.5 (using new function headers)
- requirements.txt lists all dependencies
- we recommend using anaconda distribution

Required
- [Tensorflow Installation](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation) depending on OS
- Install requirements (NumPy, pandas, SciPy, ...)
- `pip install -r requirements.txt`

Confirm installation by running `python -m unittest discover` or `python3 -m unittest discover`

Modify/run IPython notebook files using e.g. `jupyter notebook features.ipynb`
