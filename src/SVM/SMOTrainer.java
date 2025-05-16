
package SVM;

import java.util.Random;

public class SMOTrainer {

    public enum KernelType {
        LINEAR,
        RBF
    }

    private double[][] X;
    private int[] y;
    private double[] alphas;
    private double b;
    private double C = 1.0;
    private double tol = 1e-3;
    private double eps = 1e-3;
    private int maxPasses = 50; // Increased from 5 to 50 for better convergence
    private double gamma = 0.05; // RBF kernel parameter

    private int m;
    private int n;

    private Random rand = new Random();

    private KernelType kernelType = KernelType.LINEAR;

    public SMOTrainer(double[][] X, int[] y, double C) {
        this(X, y, C, KernelType.LINEAR, 0.05);
    }

    public SMOTrainer(double[][] X, int[] y, double C, KernelType kernelType, double gamma) {
        this.X = X;
        this.y = y;
        this.C = C;
        this.m = X.length;
        this.n = X[0].length;
        this.alphas = new double[m];
        this.b = 0.0;
        this.kernelType = kernelType;
        this.gamma = gamma;
    }

    private double kernel(double[] x, double[] z) {
        if (kernelType == KernelType.LINEAR) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += x[i] * z[i];
            }
            return dot;
        } else if (kernelType == KernelType.RBF) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = x[i] - z[i];
                sum += diff * diff;
            }
            return Math.exp(-gamma * sum);
        }
        return 0.0;
    }

    private double f(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            if (alphas[i] > 0) {
                sum += alphas[i] * y[i] * kernel(X[i], x);
            }
        }
        return sum + b;
    }

    public void train() {
        int passes = 0;
        double[] errors = new double[m];
        for (int i = 0; i < m; i++) {
            errors[i] = f(X[i]) - y[i];
        }

        while (passes < maxPasses) {
            int numChangedAlphas = 0;
            for (int i = 0; i < m; i++) {
                double Ei = errors[i];
                if ((y[i] * Ei < -tol && alphas[i] < C) || (y[i] * Ei > tol && alphas[i] > 0)) {
                    int j = selectJ(i, errors);
                    double Ej = errors[j];

                    double alphaIold = alphas[i];
                    double alphaJold = alphas[j];

                    double L, H;
                    if (y[i] != y[j]) {
                        L = Math.max(0, alphas[j] - alphas[i]);
                        H = Math.min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = Math.max(0, alphas[i] + alphas[j] - C);
                        H = Math.min(C, alphas[i] + alphas[j]);
                    }
                    if (L == H) continue;

                    double eta = 2 * kernel(X[i], X[j]) - kernel(X[i], X[i]) - kernel(X[j], X[j]);
                    if (eta >= 0) continue;

                    alphas[j] = alphas[j] - (y[j] * (Ei - Ej)) / eta;
                    alphas[j] = Math.max(L, Math.min(H, alphas[j]));

                    if (Math.abs(alphas[j] - alphaJold) < eps) continue;

                    alphas[i] = alphas[i] + y[i] * y[j] * (alphaJold - alphas[j]);

                    double b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * kernel(X[i], X[i])
                            - y[j] * (alphas[j] - alphaJold) * kernel(X[i], X[j]);
                    double b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * kernel(X[i], X[j])
                            - y[j] * (alphas[j] - alphaJold) * kernel(X[j], X[j]);

                    if (0 < alphas[i] && alphas[i] < C) {
                        b = b1;
                    } else if (0 < alphas[j] && alphas[j] < C) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2.0;
                    }

                    numChangedAlphas++;

                    for (int k = 0; k < m; k++) {
                        errors[k] = f(X[k]) - y[k];
                    }
                }
            }
            System.out.println("Pass " + passes + ": Number of alpha changes = " + numChangedAlphas);
            passes = (numChangedAlphas == 0) ? passes + 1 : 0;
        }
    }

    private int selectJ(int i, double[] errors) {
        int j = -1;
        double maxDelta = 0;
        for (int k = 0; k < m; k++) {
            if (k == i) continue;
            double delta = Math.abs(errors[i] - errors[k]);
            if (delta > maxDelta) {
                maxDelta = delta;
                j = k;
            }
        }
        if (j == -1) {
            j = rand.nextInt(m);
            while (j == i) {
                j = rand.nextInt(m);
            }
        }
        return j;
    }

    public double[] getAlphas() {
        return alphas;
    }

    public double getB() {
        return b;
    }
}
