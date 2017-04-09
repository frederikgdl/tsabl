package sa_embedding;

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

        List<String> posFiles = new ArrayList<String>();
        List<String> negFiles = new ArrayList<String>();
        List<String> neuFiles = new ArrayList<String>();
//        for(int i = 0; i < trainFileNum; i++)
//        {
//            posFiles.add(inputDir + "emoticon.pos." + i + ".txt");
//            negFiles.add(inputDir + "emoticon.neg." + i + ".txt");
//            neuFiles.add(inputDir + "emoticon.neu." + i + ".txt");
//        }
        posFiles.add(inputDir + inputFilePrefix + ".pos.txt");
        negFiles.add(inputDir + inputFilePrefix + ".neg.txt");
        neuFiles.add(inputDir + inputFilePrefix + ".neu.txt");

        List<String> allTrainFiles = new ArrayList<String>();
        allTrainFiles.addAll(posFiles);
        allTrainFiles.addAll(negFiles);
        allTrainFiles.addAll(neuFiles);

        HashMap<String, Integer> vocabMap  = new HashMap<String, Integer>();

        // Funcs.getVocab(vocabFile, vocabMap, "utf8");
        Funcs.getVocab(allTrainFiles, "utf8", vocabMap, 5);
        System.out.println("vocab.size(): " + vocabMap.size());

        TernaryHybridRankingMain posMain = new TernaryHybridRankingMain(
                xWindowSize, vocabMap.size(), xHiddenSize, xEmbeddingLength);

        Random rnd = new Random();
        posMain.randomize(rnd, -randomBase, randomBase);

        TernaryHybridRankingMain negMain = posMain.cloneWithTiedParams();

        double lossV = 0.0;
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

                        posMain.forward();
                        negMain.forward();

                        lossC += 1;

                        for(int k = 0; k < posMain.sentimentLinear2.outputLength; k++)
                        {
                            posMain.sentimentLinear2.outputG[k] = 0;
                            negMain.sentimentLinear2.outputG[k] = 0;// remain this part as zero.
                        }

                        if(posMain.sentimentLinear2.output[data.goldPol]
                                < posMain.sentimentLinear2.output[(data.goldPol + 1) % 3]
                                + posMain.sentimentLinear2.output[(data.goldPol + 2) % 3] + margin)
                        {
                            lossV += sentimentAlpha * (margin + posMain.sentimentLinear2.output[(data.goldPol + 1) % 3]
                                    + posMain.sentimentLinear2.output[(data.goldPol + 2) % 3]
                                    - posMain.sentimentLinear2.output[data.goldPol]);

                            posMain.sentimentLinear2.outputG[data.goldPol] = sentimentAlpha * 1;
                            posMain.sentimentLinear2.outputG[(data.goldPol + 1) % 3] = sentimentAlpha * -1;
                            posMain.sentimentLinear2.outputG[(data.goldPol + 2) % 3] = sentimentAlpha * -1;
                        }

                        // loss function
                        if(posMain.contextLinear2.output[0] < negMain.contextLinear2.output[0] + margin)
                        {
                            lossV += (1-sentimentAlpha) * (margin + negMain.contextLinear2.output[0] - posMain.contextLinear2.output[0]);
                            posMain.contextLinear2.outputG[0] = (1 - sentimentAlpha) * 1;
                            negMain.contextLinear2.outputG[0] = (1 - sentimentAlpha) * -1;
                        }

                        posMain.backward();
                        negMain.backward();

                        posMain.update(learningRate);
                        negMain.update(learningRate);

                        posMain.clearGrad();
                        negMain.clearGrad();
                    }

                    if(dataIdx % 50000 == 0)
                    {
                        System.out.println("running " + dataIdx + "/" + trainingDatas.size() +
                                "\t loss: " + (lossV / lossC) + "\t" + DateFormat.getDateTimeInstance().format(new Date()));
                    }
                }

                trainingDatas.clear();
            }

            Funcs.dumpEmbedFile(outputFile + "-round-" + round,
                    "utf8", vocabMap, posMain.lookup.table, xEmbeddingLength);
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
