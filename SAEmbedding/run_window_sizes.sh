#!/usr/bin/env bash

HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
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

for WINDOW_SIZE in 3 5 7 9
do
    OUTPUT_FILE=${OUTPUT_DIR}'windowsize-'${WINDOW_SIZE}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    echo "Running with window size "${WINDOW_SIZE}

    java -classpath bin sa_embedding.AggTernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done