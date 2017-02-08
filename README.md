# tsabl
Twitter Sentiment Analysis

## Setup
1. Make sure you have python 3, pip and virtualenv installed.
2. Go to [https://apps.twitter.com/](https://apps.twitter.com/), log in, and create a new app to get a **consumer key** and a **consumer secret**.
3. Perform the following steps (remember to insert your key and secret):

```bash
git clone https://github.com/frederikgdl/tsabl.git
cd tsabl
virtualenv venv -p python3
echo 'export TSABL_CONSUMER_KEY="<YOUR CONSUMER KEY HERE>"' >> venv/bin/activate
echo 'export TSABL_CONSUMER_SECRET="<YOUR CONSUMER KEY SECRET>"' >> venv/bin/activate
echo 'export PYTHONPATH="$PYTHONPATH:$PWD/tsabl"' >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
```
