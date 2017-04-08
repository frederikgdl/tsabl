#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
INPUT_DIR=../data/preprocessed/
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
LEARNING_RATE=0.1
MARGIN=1
OUTPUT_FILE=../data/embeddings/sa_embedding_output
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

DATA_DIR='/data/twitty/tsabl/preprocessed/datasets/1M/'

DATA_PREFIX='tweets.'

for DATASET in 'AFINN' 'EMOTICON_EXT' 'EMOTICON.150K' 'TEXTBLOB' 'VADER'

java -classpath bin sa_embedding.HybridRankingMain -windowSize ${WINDOW_SIZE} \
 -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
 -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
 -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
 -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
 -sentimentAlpha ${SENTIMENT_ALPHA}
