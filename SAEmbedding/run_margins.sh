#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
LEARNING_RATE=0.1
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/ternary/'

DATA_PREFIX='tweets.'

DATASET='LexiconClassifier'

INPUT_PREFIX=${DATA_PREFIX}${DATASET}

for MARGIN in 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 2.0 3.0 4.0 5.0 10.0
do
    OUTPUT_FILE=${OUTPUT_DIR}'margin-'${MARGIN}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    echo "Running with margin "${MARGIN}

    java -classpath bin ternary_embedding.TernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done
