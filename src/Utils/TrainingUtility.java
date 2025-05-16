package Utils;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import Hog.HogExtractor;

public class TrainingUtility {

    private static final int TRAINING_IMAGE_WIDTH = 128;
    private static final int TRAINING_IMAGE_HEIGHT = 128;

    public static BufferedImage resizeImage(BufferedImage original, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    public static double[][] normalizeFeatures(double[][] features) {
        int numSamples = features.length;
        int numFeatures = features[0].length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < numSamples; i++) {
                sum += features[i][j];
            }
            means[j] = sum / numSamples;
        }

        for (int j = 0; j < numFeatures; j++) {
            double sumSq = 0.0;
            for (int i = 0; i < numSamples; i++) {
                double diff = features[i][j] - means[j];
                sumSq += diff * diff;
            }
            stds[j] = Math.sqrt(sumSq / numSamples);
            if (stds[j] == 0) {
                stds[j] = 1.0;
            }
        }

        double[][] normalized = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                normalized[i][j] = (features[i][j] - means[j]) / stds[j];
            }
        }

        return normalized;
    }

    public static Object[] loadTrainingData(String positiveFolderPath, String negativeFolderPath) throws IOException {
        List<double[]> featureList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        HogExtractor hogExtractor = new HogExtractor();

        File posFolder = new File(positiveFolderPath);
        File[] posFiles = posFolder.listFiles();
        if (posFiles == null) {
            throw new IOException("Positive training folder is empty or does not exist.");
        }
        File resizedPosFolder = new File(positiveFolderPath, "TrainImages/positive");
        if (!resizedPosFolder.exists()) {
            resizedPosFolder.mkdirs();
        }

        for (File file : posFiles) {
            if (file.isFile() && isImageFile(file)) {
                BufferedImage img = readAndConvertImage(file);
                if (img == null) continue;
                double[] features = hogExtractor.extractHOG(img);
                featureList.add(features);
                labelList.add(1);
            }
        }

        File negFolder = new File(negativeFolderPath);
        File[] negFiles = negFolder.listFiles();
        if (negFiles == null) {
            throw new IOException("Negative training folder is empty or does not exist.");
        }
        File resizedNegFolder = new File(negativeFolderPath, "TrainImages/negative");
        if (!resizedNegFolder.exists()) {
            resizedNegFolder.mkdirs();
        }

        for (File file : negFiles) {
            if (file.isFile() && isImageFile(file)) {
                BufferedImage img = readAndConvertImage(file);
                if (img == null) continue;
                double[] features = hogExtractor.extractHOG(img);
                featureList.add(features);
                labelList.add(-1);
            }
        }

        double[][] featuresArray = featureList.toArray(new double[0][]);
        int[] labelsArray = labelList.stream().mapToInt(i -> i).toArray();

        double[][] normalizedFeatures = normalizeFeatures(featuresArray);

        return new Object[]{ normalizedFeatures, labelsArray };
    }

    private static boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".bmp");
    }

    private static BufferedImage readAndConvertImage(File file) {
        try {
            BufferedImage img = ImageIO.read(file);
            if (img == null) {
                System.err.println("Warning: Could not read image file " + file.getName() + ", skipping.");
                return null;
            }
            BufferedImage rgbImg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g2d = rgbImg.createGraphics();
            g2d.drawImage(img, 0, 0, null);
            g2d.dispose();
            return rgbImg;
        } catch (IOException e) {
            System.err.println("Warning: Exception reading image file " + file.getName() + ": " + e.getMessage() + ", skipping.");
            return null;
        }
    }
}
