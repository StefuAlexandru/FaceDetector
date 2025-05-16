package Train;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Hog.HogFeatureExtractor;
import SVM.SVMTrainer;

public class MultiPersonSVMTrainer {

    private Map<String, List<float[]>> personFeatures;

    public MultiPersonSVMTrainer() {
        personFeatures = new HashMap<>();
    }


    public void trainAndSaveForUser(String baseDir, String userName) throws IOException {
        File baseFolder = new File(baseDir);
        if (!baseFolder.exists() || !baseFolder.isDirectory()) {
            throw new IOException("Invalid base directory: " + baseDir);
        }

        HogFeatureExtractor hogExtractor = new HogFeatureExtractor();
        System.out.println("Extracting features from images for user: " + userName);
        List<double[]> featuresDouble = hogExtractor.extractFeaturesFromDirectory(baseDir + File.separator + userName);
        List<float[]> featuresFloat = new ArrayList<>();
        for (double[] feat : featuresDouble) {
            float[] ffeat = new float[feat.length];
            for (int i = 0; i < feat.length; i++) {
                ffeat[i] = (float) feat[i];
            }
            featuresFloat.add(ffeat);
        }
        personFeatures.put(userName, featuresFloat);
        
        List<float[]> trainingDataList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();

        List<float[]> positiveFeatures = personFeatures.get(userName);
        trainingDataList.addAll(positiveFeatures);
        for (int i = 0; i < positiveFeatures.size(); i++) {
            labelsList.add(1);
        }
        System.out.println("Positive samples for " + userName + ": " + positiveFeatures.size());

        for (File personDir : baseFolder.listFiles(File::isDirectory)) {
            String otherPerson = personDir.getName();
            if (!otherPerson.equals(userName)) {
                System.out.println("Processing negative samples for " + otherPerson);

                List<double[]> featuresDoubleOtherPerson = hogExtractor.extractFeaturesFromDirectory(personDir.getAbsolutePath());

                List<float[]> negativeFeatures = new ArrayList<>();
                for (double[] feat : featuresDoubleOtherPerson) {
                    float[] ffeat = new float[feat.length];
                    for (int i = 0; i < feat.length; i++) {
                        ffeat[i] = (float) feat[i];
                    }
                    negativeFeatures.add(ffeat);
                }

                trainingDataList.addAll(negativeFeatures);
                
                for (int i = 0; i < negativeFeatures.size(); i++) {
                    labelsList.add(-1);
                }
            }
        }
        System.out.println("Negative samples for " + userName + ": " + (trainingDataList.size() - positiveFeatures.size()));
        System.out.println("Total samples for " + userName + ": " + trainingDataList.size());
        float[][] trainingData = trainingDataList.toArray(new float[trainingDataList.size()][]);
        int[] labels = labelsList.stream().mapToInt(Integer::intValue).toArray();

        SVMTrainer trainer = new SVMTrainer();
        System.out.println(trainingData.length + " samples, " + trainingData[0].length + " features.");
        trainer.train(trainingData, labels);
        
        saveClassifierForUser(userName, trainer);
        System.out.println("Trained and saved classifier for user: " + userName);
    }

    public void saveClassifierForUser(String userName, SVMTrainer trainer) throws IOException {
        File userDir = new File("images" + File.separator + userName + File.separator + "userModel");
        if (!userDir.exists()) {
            userDir.mkdirs();
        }

        String modelPath = userDir.getAbsolutePath() + File.separator + userName + "Model.model";
        trainer.saveModel(modelPath);
        System.out.println("Model pentru " + userName + " salvat în: " + modelPath);
    }

}
