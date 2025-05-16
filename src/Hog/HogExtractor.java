package Hog;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

public class HogExtractor {

    private int cellSize = 8;
    private int blockSize = 2;
    private int bins = 9;
    private boolean gammaCorrection = true;
    private boolean useGaussianBlur = true;
    private boolean useL2Hys = true;

    private double[][] grayscaleImage;
    private double[][] gradientMagnitudes;
    private double[][] gradientAngles;

    public HogExtractor() {}

    public HogExtractor(int cellSize, int blockSize, int bins) {
        this.cellSize = cellSize;
        this.blockSize = blockSize;
        this.bins = bins;
    }

    public void setGammaCorrection(boolean value) {
        this.gammaCorrection = value;
    }

    public void setUseGaussianBlur(boolean value) {
        this.useGaussianBlur = value;
    }

    public void setUseL2Hys(boolean value) {
        this.useL2Hys = value;
    }

    public double[] extractHOG(BufferedImage faceImage) {
        BufferedImage image = resizeImage(faceImage, 128, 128);

        if (gammaCorrection) {
            image = applyGammaCorrection(image);
        }

        grayscaleImage = convertToGrayscale(image);

        if (useGaussianBlur) {
            grayscaleImage = applyGaussianBlur(grayscaleImage);
        }

        computeGradients();

        return computeHOGDescriptor();
    }

    private BufferedImage resizeImage(BufferedImage image, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private BufferedImage applyGammaCorrection(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage corrected = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color c = new Color(image.getRGB(x, y));
                int r = (int) (255 * Math.pow(c.getRed() / 255.0, 0.5));
                int g = (int) (255 * Math.pow(c.getGreen() / 255.0, 0.5));
                int b = (int) (255 * Math.pow(c.getBlue() / 255.0, 0.5));
                corrected.setRGB(x, y, new Color(r, g, b).getRGB());
            }
        }
        return corrected;
    }

    private double[][] convertToGrayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] gray = new double[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color c = new Color(image.getRGB(x, y));
                gray[y][x] = 0.299 * c.getRed() + 0.587 * c.getGreen() + 0.114 * c.getBlue();
            }
        }
        return gray;
    }

    private double[][] applyGaussianBlur(double[][] src) {
        int h = src.length;
        int w = src[0].length;
        double[][] dst = new double[h][w];
        double[][] kernel = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                double sum = 0;
                double weightSum = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        double val = src[y + ky][x + kx];
                        double weight = kernel[ky + 1][kx + 1];
                        sum += val * weight;
                        weightSum += weight;
                    }
                }
                dst[y][x] = sum / weightSum;
            }
        }
        return dst;
    }

    private void computeGradients() {
        int height = grayscaleImage.length;
        int width = grayscaleImage[0].length;

        gradientMagnitudes = new double[height][width];
        gradientAngles = new double[height][width];

        int[][] gx = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
        int[][] gy = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gradX = 0, gradY = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    int py = Math.min(Math.max(y + ky, 0), height - 1);
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = Math.min(Math.max(x + kx, 0), width - 1);
                        double pixel = grayscaleImage[py][px];
                        gradX += gx[ky + 1][kx + 1] * pixel;
                        gradY += gy[ky + 1][kx + 1] * pixel;
                    }
                }

                gradX /= 8.0;
                gradY /= 8.0;

                gradientMagnitudes[y][x] = Math.hypot(gradX, gradY);
                double angle = Math.toDegrees(Math.atan2(gradY, gradX));
                if (angle < 0) angle += 180;
                gradientAngles[y][x] = angle;
            }
        }
    }

    private double[] computeHOGDescriptor() {
        int height = grayscaleImage.length;
        int width = grayscaleImage[0].length;

        int cellsX = width / cellSize;
        int cellsY = height / cellSize;

        double[][][] cellHistograms = new double[cellsY][cellsX][bins];
        double binWidth = 180.0 / bins;

        for (int cy = 0; cy < cellsY; cy++) {
            for (int cx = 0; cx < cellsX; cx++) {
                for (int y = 0; y < cellSize; y++) {
                    for (int x = 0; x < cellSize; x++) {
                        int posX = cx * cellSize + x;
                        int posY = cy * cellSize + y;

                        if (posX >= width || posY >= height) continue;

                        double magnitude = gradientMagnitudes[posY][posX];
                        double angle = gradientAngles[posY][posX];

                        double bin = angle / binWidth;
                        int binLow = (int) Math.floor(bin) % bins;
                        int binHigh = (binLow + 1) % bins;
                        double weightHigh = bin - binLow;
                        double weightLow = 1.0 - weightHigh;

                        cellHistograms[cy][cx][binLow] += magnitude * weightLow;
                        cellHistograms[cy][cx][binHigh] += magnitude * weightHigh;
                    }
                }
            }
        }

        int blocksX = cellsX - blockSize + 1;
        int blocksY = cellsY - blockSize + 1;
        int blockHistogramSize = blockSize * blockSize * bins;

        double[] hogVector = new double[blocksX * blocksY * blockHistogramSize];
        int vectorIndex = 0;

        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {
                double[] block = new double[blockHistogramSize];
                int index = 0;

                for (int dy = 0; dy < blockSize; dy++) {
                    for (int dx = 0; dx < blockSize; dx++) {
                        double[] hist = cellHistograms[by + dy][bx + dx];
                        for (double val : hist) {
                            block[index++] = val;
                        }
                    }
                }

                double norm = 0;
                for (double v : block) norm += v * v;
                norm = Math.sqrt(norm + 1e-6);

                for (int i = 0; i < block.length; i++) {
                    block[i] /= norm;
                    if (useL2Hys && block[i] > 0.2) block[i] = 0.2;
                }

                if (useL2Hys) {
                    norm = 0;
                    for (double v : block) norm += v * v;
                    norm = Math.sqrt(norm + 1e-6);
                    for (int i = 0; i < block.length; i++) {
                        block[i] /= norm;
                    }
                }

                System.arraycopy(block, 0, hogVector, vectorIndex, block.length);
                vectorIndex += block.length;
            }
        }

        return hogVector;
    }
}
