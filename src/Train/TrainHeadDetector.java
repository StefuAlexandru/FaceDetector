package Train;

import java.io.IOException;

import SVM.SVMClassifier;
import Utils.TrainingUtility;

public class TrainHeadDetector {

    public static void main(String[] args) {
        String positiveFolder = "c:/Users/Alex/Desktop/Proiect_AI/src/TrainImages/positive"; // Folder with positive training images (heads)
        String negativeFolder = "c:/Users/Alex/Desktop/Proiect_AI/src/TrainImages/negative"; // Folder with negative training images (non-heads)

        double C = 1.0;
        SVMClassifier.KernelType kernelType = SVMClassifier.KernelType.LINEAR;
        double gamma = 0.05;

        // Parse command line arguments for C, kernelType, gamma
        if (args.length >= 1) {
            try {
                C = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Invalid C parameter, using default 1.0");
            }
        }
        if (args.length >= 2) {
            if (args[1].equalsIgnoreCase("RBF")) {
                kernelType = SVMClassifier.KernelType.RBF;
            } else if (args[1].equalsIgnoreCase("LINEAR")) {
                kernelType = SVMClassifier.KernelType.LINEAR;
            } else {
                System.err.println("Invalid kernel type, using default LINEAR");
            }
        }
        if (args.length >= 3) {
            try {
                gamma = Double.parseDouble(args[2]);
            } catch (NumberFormatException e) {
                System.err.println("Invalid gamma parameter, using default 0.05");
            }
        }

        try {
            // Load training data: features and labels
            Object[] data = TrainingUtility.loadTrainingData(positiveFolder, negativeFolder);
            double[][] features = (double[][]) data[0];
            int[] labels = (int[]) data[1];

            System.out.println("Loaded " + features.length + " training samples.");

            // Create and train SVM classifier
            SVMClassifier svm = new SVMClassifier();
            svm.train(features, labels, C, kernelType, gamma);

            // Debug: print alphas and bias
            double[] alphas = svm.getAlphas();
            double b = svm.getB();
            System.out.println("Trained model parameters:");
            System.out.println("Bias (b): " + b);
            System.out.println("Alphas:");
            for (int i = 0; i < alphas.length; i++) {
                if (alphas[i] > 1e-6) {
                    System.out.println("Alpha[" + i + "] = " + alphas[i] + ", Label = " + labels[i]);
                }
            }

            // Save the trained model
            try {
                svm.saveModel("trained_head_detector.model");
                System.out.println("Model saved to trained_head_detector.model");
            } catch (IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
            }

            System.out.println("Training completed successfully.");

        } catch (IOException e) {
            System.err.println("Error loading training data: " + e.getMessage());
        }
    }
}
