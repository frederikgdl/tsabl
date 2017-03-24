#!/bin/bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
INPUT_DIR=data/filtered/
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=3
LEARNING_RATE=0.1
MARGIN=1
OUTPUT_FILE=saembedding_output.txt
RANDOM_BASE=420
SENTIMENT_ALPHA=0.5

java -classpath bin sa_embedding.HybridRankingMain -windowSize ${WINDOW_SIZE} \
 -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
 -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
 -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
 -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
 -sentimentAlpha ${SENTIMENT_ALPHA}
