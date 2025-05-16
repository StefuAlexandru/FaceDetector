package Utils;

import Hog.HogExtractor;
import SVM.SVMTrainer;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

public class SVMTester {

    private SVMTrainer svmTrainer;
    private HogExtractor hogExtractor;
    private static final int WINDOW_SIZE = 128;
    private static final int STEP_SIZE = 32;

    public SVMTester() {
        svmTrainer = new SVMTrainer();
        hogExtractor = new HogExtractor();
    }

    public void loadModel(String modelPath) {
        svmTrainer.loadModel(modelPath);
    }

    public List<Rectangle> getDetections(BufferedImage inputImage) {
        List<Rectangle> detections = new ArrayList<>();
        double scaleFactor = 1.0;
        double minScale = 0.6;

        while (scaleFactor >= minScale) {
            BufferedImage scaledImage = resizeImage(inputImage, scaleFactor);
            int scaledWidth = scaledImage.getWidth();
            int scaledHeight = scaledImage.getHeight();
            System.out.println("Scaled image size: " + scaledWidth + "x" + scaledHeight);
            for (int y = 0; y <= scaledHeight - WINDOW_SIZE; y += STEP_SIZE) {
                for (int x = 0; x <= scaledWidth - WINDOW_SIZE; x += STEP_SIZE) {
                    BufferedImage window = scaledImage.getSubimage(x, y, WINDOW_SIZE, WINDOW_SIZE);
                    double[] hogFeatures = hogExtractor.extractHOG(window);

                    float[] featuresFloat = new float[hogFeatures.length];
                    for (int i = 0; i < hogFeatures.length; i++) {
                        featuresFloat[i] = (float) hogFeatures[i];
                    }

                    float score = svmTrainer.predict(featuresFloat);
                    if (score > 0.0f) {
                        System.out.println("Detected head at: " + x + ", " + y + " with scale: " + scaleFactor);
                        int originalX = (int) (x / scaleFactor);
                        int originalY = (int) (y / scaleFactor);
                        int originalSize = (int) (WINDOW_SIZE / scaleFactor);
                        detections.add(new Rectangle(originalX, originalY, originalSize, originalSize));
                    }
                }
            }

            scaleFactor -= 0.1;
        }

        return clusterDetections(detections, 150,128*128);
    }

    public BufferedImage detectFaces(BufferedImage inputImage) {
        BufferedImage annotatedImage = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), inputImage.getType());
        Graphics2D g2d = annotatedImage.createGraphics();
        g2d.drawImage(inputImage, 0, 0, null);
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new java.awt.BasicStroke(2));

        List<Rectangle> finalRects = getDetections(inputImage);

        for (Rectangle rect : finalRects) {
            g2d.drawRect(rect.x, rect.y, rect.width, rect.height);
        }

        g2d.dispose();
        return annotatedImage;
    }

    private BufferedImage resizeImage(BufferedImage originalImage, double scale) {
        int newWidth = (int) (originalImage.getWidth() * scale);
        int newHeight = (int) (originalImage.getHeight() * scale);
        BufferedImage resized = new BufferedImage(newWidth, newHeight, originalImage.getType());
        Graphics2D g = resized.createGraphics();
        g.drawImage(originalImage, 0, 0, newWidth, newHeight, null);
        g.dispose();
        return resized;
    }

    private List<Rectangle> clusterDetections(List<Rectangle> rects, double distanceThreshold, int minArea) {
        List<Rectangle> clustered = new ArrayList<>();
        boolean[] visited = new boolean[rects.size()];

        for (int i = 0; i < rects.size(); i++) {
            if (visited[i]) continue;

            Rectangle base = rects.get(i);
            List<Rectangle> cluster = new ArrayList<>();
            cluster.add(base);
            visited[i] = true;

            for (int j = i + 1; j < rects.size(); j++) {
                if (visited[j]) continue;

                Rectangle other = rects.get(j);
                double dx = base.getCenterX() - other.getCenterX();
                double dy = base.getCenterY() - other.getCenterY();
                double distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < distanceThreshold) {
                    visited[j] = true;
                    cluster.add(other);
                }
            }

            int minX = Integer.MAX_VALUE;
            int minY = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE;
            int maxY = Integer.MIN_VALUE;

            for (Rectangle r : cluster) {
                minX = Math.min(minX, r.x);
                minY = Math.min(minY, r.y);
                maxX = Math.max(maxX, r.x + r.width);
                maxY = Math.max(maxY, r.y + r.height);
            }

            Rectangle finalCluster = new Rectangle(minX, minY, maxX - minX, maxY - minY);
            clustered.add(finalCluster);
            
        }

        return clustered;
    }


    public void detectFacesAndSave(String inputImagePath) throws IOException {
        BufferedImage inputImage = ImageIO.read(new File(inputImagePath));
        BufferedImage annotatedImage = detectFaces(inputImage);
        String outputImagePath = "face_Detected.png";
        ImageIO.write(annotatedImage, "png", new File(outputImagePath));
        System.out.println("Face detection completed. Output saved to " + outputImagePath);
    }

    public void detectAndSaveAllHeads(String inputImagePath) throws IOException {
        BufferedImage inputImage = ImageIO.read(new File(inputImagePath));
        List<Rectangle> detections = getDetections(inputImage);

        if (detections.isEmpty()) {
            System.out.println("No heads detected.");
            return;
        }

        int count = 0;
        for (Rectangle rect : detections) {
            int x = Math.max(0, rect.x);
            int y = Math.max(0, rect.y);
            int width = Math.min(rect.width, inputImage.getWidth() - x);
            int height = Math.min(rect.height, inputImage.getHeight() - y);

            if (width <= 0 || height <= 0) continue;

            BufferedImage cropped = inputImage.getSubimage(x, y, width, height);

            BufferedImage resized = new BufferedImage(128, 128, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resized.createGraphics();
            g.drawImage(cropped, 0, 0, 128, 128, null);
            g.dispose();

            String filename = String.format("head_%d.png", count++);
            ImageIO.write(resized, "png", new File(filename));
        }

        System.out.println("Saved " + count + " head(s) to disk.");
    }

    public static void main(String[] args) {
        String modelPath = "Model/trained_head_detector.model";
        String inputImagePath;

        if (args.length == 1) {
            inputImagePath = args[0];
        } else if (args.length >= 2) {
            modelPath = args[0];
            inputImagePath = args[1];
        } else {
            System.out.println("Usage: java SVMTester [modelPath] imagePath");
            return;
        }

        SVMTester tester = new SVMTester();
        tester.loadModel(modelPath);

        try {
            // Folosește această metodă pentru a salva toate capetele în fișiere separate:
            //tester.detectAndSaveAllHeads(inputImagePath);

            // Sau această metodă doar pentru a desena pătratele pe imagine:
            tester.detectFacesAndSave(inputImagePath);

        } catch (IOException e) {
            System.err.println("Error processing images: " + e.getMessage());
        }
    }
}
