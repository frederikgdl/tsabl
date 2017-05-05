#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
LEARNING_RATE=0.1
MARGIN=1
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/agg_ternary/'

DATA_PREFIX='tweets.'

DATASET='LexiconClassifier'

INPUT_PREFIX=${DATA_PREFIX}${DATASET}

for EMBEDDING_LENGTH in 50 75 100 125 150
do
    OUTPUT_FILE=${OUTPUT_DIR}'embeddinglen-'${EMBEDDING_LENGTH}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    echo "Running with embedding length "${EMBEDDING_LENGTH}

    java -classpath bin sa_embedding.AggTernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done