package Utils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.opencv.ml.SVM;

public class ModelLoader {

    static {
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }
    public static Map<String, SVM> loadAllUserModels(String basePath) {
        Map<String, SVM> userModels = new HashMap<>();

        File baseDir = new File(basePath);
        if (!baseDir.exists() || !baseDir.isDirectory()) {
            System.err.println("Base path does not exist or is not a directory: " + basePath);
            return userModels;
        }

        File[] userDirs = baseDir.listFiles(File::isDirectory);
        if (userDirs == null) return userModels;

        for (File userDir : userDirs) {
            String userName = userDir.getName();
            File modelFile = new File(userDir, "userModel" + File.separator + userName + "Model.model");

            if (modelFile.exists()) {
                try {
                    SVM model = SVM.load(modelFile.getAbsolutePath());
                    userModels.put(userName, model);
                    System.out.println("Loaded model for user: " + userName);
                } catch (Exception e) {
                    System.err.println("Failed to load model for " + userName + ": " + e.getMessage());
                }
            } else {
                System.out.println("No model found for user: " + userName);
            }
        }

        return userModels;
    }
}
