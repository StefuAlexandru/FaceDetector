package Train;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import Hog.HogExtractor;
import SVM.SVMTrainer;
public class TrainModel {
    
    public static void main(String[] args) {
        System.out.println("=== Starting SVM Training Process ===");

        float[][] features = loadFeatures();
        int[] labels = loadLabels();

        System.out.println("Training data prepared.");
        System.out.println("Training SVM...");

        SVMTrainer trainer = new SVMTrainer();
        trainer.train(features, labels);

        System.out.println("SVM training completed.");

        String modelPath = "Model/trained_head_detector.model";
        trainer.saveModel(modelPath);
        System.out.println("Model saved to: " + modelPath);

        System.out.println("Training process completed successfully");
    }

    private static float[][] loadFeatures() {
        ArrayList<float[]> featureList = new ArrayList<>();
        HogExtractor hog = new HogExtractor();

        File positiveHogFile = new File("TrainImages/positiveHog.txt");
        File negativeHogFile = new File("TrainImages/negativeHog.txt");

        try (java.io.PrintWriter posWriter = new java.io.PrintWriter(positiveHogFile);
            java.io.PrintWriter negWriter = new java.io.PrintWriter(negativeHogFile)) {

            File posDir = new File("TrainImages/positive");
            if (posDir.exists() && posDir.isDirectory()) {
                for (File imgFile : posDir.listFiles()) {
                    try {
                        BufferedImage img = ImageIO.read(imgFile);
                        double[] hogFeatures = hog.extractHOG(img);

                        posWriter.println("Vector " + (featureList.size() + 1) + ":");
                        posWriter.print("[");
                        for (int i = 0; i < hogFeatures.length; i++) {
                            posWriter.printf("%.5f", hogFeatures[i]);
                            if (i < hogFeatures.length - 1) {
                                posWriter.print(", ");
                            }
                        }
                        posWriter.println("]");
                        posWriter.println();

                        float[] fFeatures = new float[hogFeatures.length];
                        for (int i = 0; i < hogFeatures.length; i++) {
                            fFeatures[i] = (float) hogFeatures[i];
                        }
                        featureList.add(fFeatures);
                    } catch (IOException e) {
                        System.err.println("Failed to read image: " + imgFile.getName());
                    }
                }
            }

            File negDir = new File("TrainImages/negative");
            if (negDir.exists() && negDir.isDirectory()) {
                for (File imgFile : negDir.listFiles()) {
                    try {
                        BufferedImage img = ImageIO.read(imgFile);
                        double[] hogFeatures = hog.extractHOG(img);

                        negWriter.println("Vector " + (featureList.size() + 1) + ":");
                        negWriter.print("[");
                        for (int i = 0; i < hogFeatures.length; i++) {
                            negWriter.printf("%.5f", hogFeatures[i]);
                            if (i < hogFeatures.length - 1) {
                                negWriter.print(", ");
                            }
                        }
                        negWriter.println("]");
                        negWriter.println();

                        float[] fFeatures = new float[hogFeatures.length];
                        for (int i = 0; i < hogFeatures.length; i++) {
                            fFeatures[i] = (float) hogFeatures[i];
                        }
                        featureList.add(fFeatures);
                    } catch (IOException e) {
                        System.err.println("Failed to read image: " + imgFile.getName());
                    }
                }
            }

        } catch (IOException e) {
            System.err.println("Failed to write HOG feature files: " + e.getMessage());
        }

        return featureList.toArray(new float[0][]);
    }


    private static int[] loadLabels() {
        ArrayList<Integer> labelList = new ArrayList<>();

        File posDir = new File("TrainImages/positive");
        if (posDir.exists() && posDir.isDirectory()) {
            int posCount = posDir.listFiles().length;
            for (int i = 0; i < posCount; i++) {
                labelList.add(1);
            }
        }

        File negDir = new File("TrainImages/negative");
        if (negDir.exists() && negDir.isDirectory()) {
            int negCount = negDir.listFiles().length;
            for (int i = 0; i < negCount; i++) {
                labelList.add(-1);
            }
        }

        int[] labels = new int[labelList.size()];
        for (int i = 0; i < labelList.size(); i++) {
            labels[i] = labelList.get(i);
        }
        return labels;
    }
}
