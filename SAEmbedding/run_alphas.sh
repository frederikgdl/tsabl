#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
LEARNING_RATE=0.1
MARGIN=1
RANDOM_BASE=0.01

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/ternary/'

DATA_PREFIX='tweets.'

DATASET='AFINN'

INPUT_PREFIX=${DATA_PREFIX}${DATASET}

for SENTIMENT_ALPHA in 0.0 0.2 0.4 0.5 0.6 0.8 1.0
do
    OUTPUT_FILE=${OUTPUT_DIR}'alpha-'${SENTIMENT_ALPHA}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    java -classpath bin sa_embedding.TernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done
