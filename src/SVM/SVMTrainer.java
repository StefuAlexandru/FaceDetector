package SVM;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;

public class SVMTrainer {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private SVM svm;

    public SVMTrainer() {
        svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.RBF);
        svm.setC(2.5);
        svm.setGamma(0.02);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 1000, 1e-6));

    }

    public void train(float[][] trainingData, int[] labels) {
        int sampleCount = trainingData.length;
        int featureCount = trainingData[0].length;

        Mat trainingMat = new Mat(sampleCount, featureCount, CvType.CV_32F);
        Mat labelsMat = new Mat(sampleCount, 1, CvType.CV_32S);

        for (int i = 0; i < sampleCount; i++) {
            for (int j = 0; j < featureCount; j++) {
                trainingMat.put(i, j, trainingData[i][j]);
            }
            labelsMat.put(i, 0, labels[i]);
        }

        svm.train(trainingMat, Ml.ROW_SAMPLE, labelsMat);

        System.out.println("Number of support vectors: " + svm.getSupportVectors().rows());
        System.out.println("Decision function rho (bias): " + svm.getDecisionFunction(0, new Mat(), new Mat()));
    }

    public float predict(float[] sample) {
        Mat sampleMat = new Mat(1, sample.length, CvType.CV_32F);
        for (int i = 0; i < sample.length; i++) {
            sampleMat.put(0, i, sample[i]);
        }
        return svm.predict(sampleMat);
    }

    public void saveModel(String filename) {
        svm.save(filename);
    }

    public void loadModel(String filename) {
        svm = SVM.load(filename);
    }
}
