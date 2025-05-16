package Hog;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class HogFeatureExtractor {

    private HogExtractor hogExtractor;

    public HogFeatureExtractor() {
        hogExtractor = new HogExtractor();
    }

    public HogFeatureExtractor(int cellSize, int blockSize, int bins) {
        hogExtractor = new HogExtractor(cellSize, blockSize, bins);
    }

    public List<double[]> extractFeaturesFromDirectory(String directoryPath) {
        List<double[]> featuresList = new ArrayList<>();
        File dir = new File(directoryPath);
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Invalid directory: " + directoryPath);
            return featuresList;
        }

        File[] imageFiles = dir.listFiles((d, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg");
        });

        if (imageFiles == null) {
            System.err.println("No image files found in directory: " + directoryPath);
            return featuresList;
        }

        for (File imgFile : imageFiles) {
            try {
                BufferedImage img = ImageIO.read(imgFile);
                if (img != null) {
                    double[] features = hogExtractor.extractHOG(img);
                    System.out.println("Extracted features from: " + imgFile.getName());
                    featuresList.add(features);
                }
            } catch (IOException e) {
                System.err.println("Failed to read image: " + imgFile.getName());
            }
        }

        return featuresList;
    }
}
