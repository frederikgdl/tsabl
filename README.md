# tsabl
Twitter Sentiment Analysis

## Setup
1. Make sure you have python 3, pip and virtualenv installed.
2. Go to [https://apps.twitter.com/](https://apps.twitter.com/), log in, and create a new app to get a **consumer key** 
and a **consumer secret**.
3. Perform the following steps (remember to insert your key and secret):
```bash
git clone https://github.com/frederikgdl/tsabl.git
cd tsabl
virtualenv venv -p python3
echo 'export TSABL_CONSUMER_KEY="<YOUR CONSUMER KEY HERE>"' >> venv/bin/activate
echo 'export TSABL_CONSUMER_SECRET="<YOUR CONSUMER KEY SECRET>"' >> venv/bin/activate
echo 'export PYTHONPATH="$PYTHONPATH:$PWD"' >> venv/bin/activate
echo 'export KERAS_BACKEND=theano' >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```

## Run
TL;DR - Do everything:
```bash
source venv/bin/activate
python embeddings/main.py
python classifiers/train.py
python classifiers/test.py
```

The `source venv/bin/activate` is needed in order to use the virtual environment and must be run before running the
 other scripts.

### Embeddings
* Edit `embeddings/config.py` to use the data files you want.
* Run `python embeddings/main.py` to train and save the embeddings.

### Classifiers
* Edit `classifiers/config.py` to use the data files you want.
* Run `python classifiers/train.py` to train and save the models.
* Run `python classifiers/test.py` to test the saved classifier models.


## Train embeddings on GPU
Embeddings can be trained on GPU using either Theano or TensorFlow backend. 
* Make sure CUDA is installed if using NVIDIA GPU. 

### Theano
1. Make sure libgpuarray is installed.
2. Edit `run_gpu.sh`: the `KERAS_BACKEND` variable must be set to `theano` and paths to libgpuarray and CUDA must be
 specified.
3. Run by typing (may need to give permission first):
```bash
./run_gpu.sh
```

### TensorFlow
1. Make sure GPU version of TensorFlow is installed.
2. Edit `run_gpu.sh`: the `KERAS_BACKEND` variable must be set to `tensorflow`.
3. Run by typing (may need to give permission first):
```bash
./run_gpu.sh
```
