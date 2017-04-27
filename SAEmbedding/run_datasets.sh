#!/usr/bin/env bash

WINDOW_SIZE=7
HIDDEN_LENGTH=20
EMBEDDING_LENGTH=50
VOCAB_FILE=FOO
TRAIN_FILE_NUM=1
TRAINING_ROUND=30
LEARNING_RATE=0.1
MARGIN=1
RANDOM_BASE=0.01
SENTIMENT_ALPHA=0.5

INPUT_DIR='../data/preprocessed/datasets/1M/'

OUTPUT_DIR='../data/embeddings/ternary/'

DATA_PREFIX='tweets.'

# 'AFINN' 'ComboA' 'ComboB' 'EMOTICON_EXT' 'EMOTICON.150K' 'LexiconClassifier' 'TEXTBLOB' 'VADER'
for DATASET in 'EMOTICON_EXT' 'EMOTICON.150K' 'LexiconClassifier' 'VADER'
do
    INPUT_PREFIX=${DATA_PREFIX}${DATASET}
    OUTPUT_FILE=${OUTPUT_DIR}${DATASET}'/embeddings-'${EMBEDDING_LENGTH}'-'${DATASET}

    java -classpath bin sa_embedding.TernaryHybridRankingMain -windowSize ${WINDOW_SIZE} \
     -hiddenLength ${HIDDEN_LENGTH} -embeddingLength ${EMBEDDING_LENGTH} \
     -inputDir ${INPUT_DIR} -vocabFile ${VOCAB_FILE} -trainFileNum ${TRAIN_FILE_NUM} \
     -trainingRound ${TRAINING_ROUND} -learningRate ${LEARNING_RATE} -margin ${MARGIN} \
     -outputFile ${OUTPUT_FILE} -randomBase ${RANDOM_BASE} \
     -sentimentAlpha ${SENTIMENT_ALPHA} -inputFilePrefix ${INPUT_PREFIX}
done
