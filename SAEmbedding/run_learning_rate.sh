#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
MARGIN=1
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/agg_ternary/'

DATA_PREFIX='tweets.'

DATASET='LexiconClassifier'

INPUT_PREFIX=${DATA_PREFIX}${DATASET}

for LEARNING_RATE in 0.05 0.2 0.3 0.5 0.7 0.9 1.1
do
    OUTPUT_FILE=${OUTPUT_DIR}'learningrate-'${LEARNING_RATE}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    echo "Running with learning rate "${LEARNING_RATE}

    java -classpath bin sa_embedding.AggTernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done