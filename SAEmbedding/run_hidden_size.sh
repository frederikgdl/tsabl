#!/usr/bin/env bash

WINDOW_SIZE=7
EMBEDDING_LENGTH=50
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=20
LEARNING_RATE=0.1
MARGIN=1
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/ternary/'

DATA_PREFIX='tweets.'

DATASET='LexiconClassifier'

INPUT_PREFIX=${DATA_PREFIX}${DATASET}

for HIDDEN_LENGTH in 10 30 50 100
do
    OUTPUT_FILE=${OUTPUT_DIR}'hiddenlength-'${HIDDEN_LENGTH}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    echo "Running with size of hidden layer "${HIDDEN_LENGTH}

    java -classpath bin ternary_embedding.TernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done
