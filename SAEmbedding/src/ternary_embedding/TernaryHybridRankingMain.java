package ternary_embedding;

import java.text.DateFormat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import funcs.Data;
import funcs.Funcs;

public class TernaryHybridRankingMain {
    public static void train(HashMap<String, String> argsMap) throws Exception
    {
        int xWindowSize = Integer.parseInt(argsMap.get("-windowSize"));
        int xHiddenSize = Integer.parseInt(argsMap.get("-hiddenLength"));
        int xEmbeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));
        String inputDir = argsMap.get("-inputDir");
        String vocabFile = argsMap.get("-vocabFile");
        int trainFileNum = Integer.parseInt(argsMap.get("-trainFileNum"));
        // 99M corresponds to 99
        int trainRound = Integer.parseInt(argsMap.get("-trainingRound"));
        double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
        double margin = Double.parseDouble(argsMap.get("-margin"));
        String outputFile = argsMap.get("-outputFile");
        double randomBase = Double.parseDouble(argsMap.get("-randomBase"));
        double sentimentAlpha = Double.parseDouble(argsMap.get("-sentimentAlpha"));
        String inputFilePrefix = argsMap.get("-inputFilePrefix");

        // Optional hyperparameters
        double dropedRatio = 0.0;
        // Boolean adaGrad = false;
        int batchsize = 1;

        if (argsMap.containsKey("-dropedRatio")) {
            dropedRatio = Double.parseDouble(argsMap.get("-dropedRatio"));
        }

//        if (argsMap.containsKey("-adaGrad")) {
//            adaGrad = Boolean.parseBoolean(argsMap.get("-adaGrad"));
//        }

        if (argsMap.containsKey("-batchsize")) {
            batchsize = Integer.parseInt(argsMap.get("-batchsize"));
        }

        List<String> posFiles = new ArrayList<String>();
        List<String> negFiles = new ArrayList<String>();
        List<String> neuFiles = new ArrayList<String>();

        posFiles.add(inputDir + inputFilePrefix + ".pos.txt");
        negFiles.add(inputDir + inputFilePrefix + ".neg.txt");
        neuFiles.add(inputDir + inputFilePrefix + ".neu.txt");

        List<String> allTrainFiles = new ArrayList<String>();
        allTrainFiles.addAll(posFiles);
        allTrainFiles.addAll(negFiles);
        allTrainFiles.addAll(neuFiles);

        HashMap<String, Integer> vocabMap  = new HashMap<String, Integer>();

        Funcs.getVocab(allTrainFiles, "utf8", vocabMap, 5);
        System.out.println("vocab.size(): " + vocabMap.size());

        TernaryHybridRanking posMain = new TernaryHybridRanking(
                xWindowSize, vocabMap.size(), xHiddenSize, xEmbeddingLength);

        Random rnd = new Random();
        posMain.randomize(rnd, -randomBase, randomBase);

        TernaryHybridRanking negMain = posMain.cloneWithTiedParams();

        double lossSent = 0.0;
        double lossContext = 0.0;
        int lossC = 0;
        for(int round = 0; round < trainRound; round++)
        {
            System.out.println("Running round: " + round);

            Collections.shuffle(posFiles);
            Collections.shuffle(negFiles);
            Collections.shuffle(neuFiles);

            for(int fileIdx = 0; fileIdx < posFiles.size(); fileIdx++)
            {
                List<Data> trainingDatas = new ArrayList<Data>();

                Funcs.readTrainFile(posFiles.get(fileIdx), "utf8",
                        0, trainingDatas);
                Funcs.readTrainFile(negFiles.get(fileIdx), "utf8",
                        1, trainingDatas);
                Funcs.readTrainFile(neuFiles.get(fileIdx), "utf8",
                        2, trainingDatas);

                System.out.println("Running pos-file: " + posFiles.get(fileIdx));
                System.out.println("Running neg-file: " + negFiles.get(fileIdx));
                System.out.println("Running neu-file: " + neuFiles.get(fileIdx));

                Collections.shuffle(trainingDatas);

                for(int dataIdx = 0; dataIdx < trainingDatas.size(); dataIdx++)
                {
                    Data data = trainingDatas.get(dataIdx);
                    if(data.words.length < xWindowSize)
                    {
                        continue;
                    }

                    for(int i = 0; i < data.words.length - xWindowSize + 1; i++)
                    {
                        int[] wordIns = Funcs.fillWindow(i, data, xWindowSize, vocabMap);
                        System.arraycopy(wordIns, 0, posMain.input, 0, xWindowSize);
                        System.arraycopy(wordIns, 0, negMain.input, 0, xWindowSize);

                        int randWordIdx = rnd.nextInt(vocabMap.size());
                        while(randWordIdx == wordIns[xWindowSize/2])
                        {
                            randWordIdx = rnd.nextInt(vocabMap.size());
                        }
                        negMain.input[xWindowSize/2] = randWordIdx;

                        // Propagate through network
                        if (dropedRatio > 0) {
                            posMain.forwardWithDropout(dropedRatio);
                            negMain.forwardWithDropout(dropedRatio);
                        } else {
                            posMain.forward();
                            negMain.forward();
                        }


                        lossC += 1;

                        for(int k = 0; k < posMain.sentimentLinear2.outputLength; k++)
                        {
                            posMain.sentimentLinear2.outputG[k] = 0;
                            negMain.sentimentLinear2.outputG[k] = 0;// remain this part as zero.
                        }

                        if(posMain.sentimentLinear2.output[data.goldPol]
                                < posMain.sentimentLinear2.output[(data.goldPol + 1) % 3] + margin)
                        {
                            lossSent += sentimentAlpha * (margin + posMain.sentimentLinear2.output[(data.goldPol + 1) % 3]
                                    - posMain.sentimentLinear2.output[data.goldPol]);

                            posMain.sentimentLinear2.outputG[data.goldPol] += sentimentAlpha * 1;
                            posMain.sentimentLinear2.outputG[(data.goldPol + 1) % 3] = sentimentAlpha * -1;
                        }

                        if(posMain.sentimentLinear2.output[data.goldPol]
                                < posMain.sentimentLinear2.output[(data.goldPol + 2) % 3] + margin)
                        {
                            lossSent += sentimentAlpha * (margin + posMain.sentimentLinear2.output[(data.goldPol + 2) % 3]
                                    - posMain.sentimentLinear2.output[data.goldPol]);

                            posMain.sentimentLinear2.outputG[data.goldPol] += sentimentAlpha * 1;
                            posMain.sentimentLinear2.outputG[(data.goldPol + 2) % 3] = sentimentAlpha * -1;
                        }

                        // loss function
                        if(posMain.contextLinear2.output[0] < negMain.contextLinear2.output[0] + margin)
                        {
                            lossContext += (1-sentimentAlpha) * (margin + negMain.contextLinear2.output[0]
                                    - posMain.contextLinear2.output[0]);
                            posMain.contextLinear2.outputG[0] = (1 - sentimentAlpha) * 1;
                            negMain.contextLinear2.outputG[0] = (1 - sentimentAlpha) * -1;
                        }

                        // Calculate gradients
                        if (dropedRatio > 0) {
                            posMain.backwardWithDropout();
                            negMain.backwardWithDropout();
                        } else {
                            posMain.backward();
                            negMain.backward();
                        }

                        // Update parameters
//                        if (adaGrad) {
//                            posMain.updateAdaGrad(learningRate, batchsize);
//                            negMain.updateAdaGrad(learningRate, batchsize);
//                        } else {
//                            posMain.update(learningRate);
//                            negMain.update(learningRate);
//                        }
                        posMain.update(learningRate);
                        negMain.update(learningRate);

                        posMain.clearGrad();
                        negMain.clearGrad();
                    }

                    if(dataIdx % 50000 == 0)
                    {
                        System.out.println("running " + dataIdx + "/" + trainingDatas.size() +
                                "\t loss: " + ((lossSent + lossContext) / lossC) + "\t context loss: " +
                                (lossContext / lossC) + "\t sentiment loss: " + (lossSent / lossC) + "\t" +
                                DateFormat.getDateTimeInstance().format(new Date()));
                    }
                }
                trainingDatas.clear();
            }
            Funcs.dumpEmbedFile(outputFile + "-round-" + round, "utf8",
                    vocabMap, posMain.lookup.table, xEmbeddingLength);
        }
    }

    public static void main(String[] args) {
        HashMap<String, String> argsMap = Funcs.parseArgs(args);

        for(String key: argsMap.keySet())
        {
            System.out.println(key + "\t" + argsMap.get(key));
        }

        try {
            train(argsMap);
            System.out.println("Done");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
