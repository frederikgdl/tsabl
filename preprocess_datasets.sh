#!/usr/bin/env bash

source venv/bin/activate

PROGRAM='twitter/tweet_preprocessor.py'

FLAGS='-tsv -l'

DATASETS_PATH='/data/twitty/data/distant_supervised/datasets/'

DATA_PREFIX='tweets.'

OUTPUT_PATH='data/preprocessed/datasets/full/'

for DATASET in 'AFINN' 'EMOTICON_EXT' 'EMOTICON' 'TEXTBLOB' 'VADER'
do
    for SUFFIX in '.pos' '.neg' '.neu'
    do
        python ${PROGRAM} ${DATASETS_PATH}${DATA_PREFIX}${DATASET}${SUFFIX}'.tsv' ${OUTPUT_PATH}${DATA_PREFIX}${DATASET}${SUFFIX}'.txt' ${FLAGS}
    done
done
