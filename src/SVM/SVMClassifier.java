package SVM;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class SVMClassifier {

    public enum KernelType {
        LINEAR,
        RBF
    }

    private double[][] supportVectors;
    private int[] supportLabels;
    private double[] alphas;
    private double b;
    private double gamma = 0.05;
    private KernelType kernelType = KernelType.LINEAR;

    public void train(double[][] X, int[] y, double C, KernelType kernelType, double gamma) {
        this.kernelType = kernelType;
        this.gamma = gamma;

        SMOTrainer trainer = new SMOTrainer(X, y, C, kernelType, gamma);
        trainer.train();

        double[] fullAlphas = trainer.getAlphas();
        List<double[]> svList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        List<Double> alphaList = new ArrayList<>();

        for (int i = 0; i < fullAlphas.length; i++) {
            if (fullAlphas[i] > 1e-6) {
                svList.add(X[i]);
                labelList.add(y[i]);
                alphaList.add(fullAlphas[i]);
            }
        }

        supportVectors = svList.toArray(new double[0][]);
        supportLabels = labelList.stream().mapToInt(Integer::intValue).toArray();
        alphas = alphaList.stream().mapToDouble(Double::doubleValue).toArray();
        b = trainer.getB();
    }

    public int predict(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < supportVectors.length; i++) {
            sum += alphas[i] * supportLabels[i] * kernel(supportVectors[i], x);
        }
        sum += b;
        return sum >= 0 ? 1 : -1;
    }

    public double decisionFunction(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < supportVectors.length; i++) {
            sum += alphas[i] * supportLabels[i] * kernel(supportVectors[i], x);
        }
        return sum + b;
    }

    private double kernel(double[] x1, double[] x2) {
        if (kernelType == KernelType.LINEAR) {
            double sum = 0.0;
            for (int i = 0; i < x1.length; i++) {
                sum += x1[i] * x2[i];
            }
            return sum;
        } else if (kernelType == KernelType.RBF) {
            double sum = 0.0;
            for (int i = 0; i < x1.length; i++) {
                double diff = x1[i] - x2[i];
                sum += diff * diff;
            }
            return Math.exp(-gamma * sum);
        }
        return 0.0;
    }

    public void saveModel(String filename) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename))) {
            out.writeObject(supportVectors);
            out.writeObject(supportLabels);
            out.writeObject(alphas);
            out.writeDouble(b);
            out.writeObject(kernelType);
            out.writeDouble(gamma);
        }
    }

    public void loadModel(String filename) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename))) {
            supportVectors = (double[][]) in.readObject();
            supportLabels = (int[]) in.readObject();
            alphas = (double[]) in.readObject();
            b = in.readDouble();
            kernelType = (KernelType) in.readObject();
            gamma = in.readDouble();
        } catch (ClassNotFoundException e) {
            throw new IOException("Class not found during model loading: " + e.getMessage());
        }
    }

    public double[] getAlphas() {
        return alphas;
    }

    public double getB() {
        return b;
    }
}
