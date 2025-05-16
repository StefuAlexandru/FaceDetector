package SVM;

import Camera.CameraCaptureGUI;
import Camera.WebcamTester;
import Train.MultiPersonSVMTrainer;
import Utils.ImageManagerGUI;
import Utils.ModelLoader;
import Utils.SVMTester;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import org.opencv.ml.SVM;

public class Main extends JFrame {
    static {
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    public Main() {
        super("Face Detection SVM Project");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new BorderLayout());

        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        JLabel numberLabel = new JLabel("Number of Photos:");
        JTextField numberField = new JTextField(10);

        JLabel photoNameLabel = new JLabel("Photo Name:");
        JTextField photoNameField = new JTextField(20);

        JButton captureButton = new JButton("Capture Photos");
        JButton manageImagesButton = new JButton("Manage Saved Images");
        JButton trainButton = new JButton("Train Classifiers");
        JButton testButton = new JButton("Test Classifiers");

        styleButton(captureButton);
        styleButton(manageImagesButton);
        styleButton(trainButton);
        styleButton(testButton);

        gbc.gridx = 0;
        gbc.gridy = 0;
        controlPanel.add(numberLabel, gbc);
        gbc.gridx = 1;
        controlPanel.add(numberField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        controlPanel.add(photoNameLabel, gbc);
        gbc.gridx = 1;
        controlPanel.add(photoNameField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.gridwidth = 2;
        controlPanel.add(captureButton, gbc);

        gbc.gridy = 3;
        controlPanel.add(manageImagesButton, gbc);

        gbc.gridy = 4;
        controlPanel.add(trainButton, gbc);

        gbc.gridy = 5;
        controlPanel.add(testButton, gbc);

        add(controlPanel, BorderLayout.CENTER);

        captureButton.addActionListener(e -> {
            String numberText = numberField.getText().trim();
            String photoName = photoNameField.getText().trim();
            if (numberText.isEmpty() || photoName.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter both the number of photos and the photo name.", "Input Required", JOptionPane.WARNING_MESSAGE);
                return;
            }
            int number;
            try {
                number = Integer.parseInt(numberText);
                if (number <= 0) {
                    JOptionPane.showMessageDialog(this, "Number of photos must be a positive integer.", "Invalid Input", JOptionPane.WARNING_MESSAGE);
                    return;
                }
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(this, "Number of photos must be a valid integer.", "Invalid Input", JOptionPane.WARNING_MESSAGE);
                return;
            }
            new CameraCaptureGUI(photoName, number);
        });

        manageImagesButton.addActionListener(e -> {
            String photoName = photoNameField.getText().trim();
            if (photoName.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter the photo name to manage images.", "Input Required", JOptionPane.WARNING_MESSAGE);
                return;
            }
            String imagesFolderPath = "images" + File.separator + photoName;
            new ImageManagerGUI(imagesFolderPath);
        });

        trainButton.addActionListener(e -> {
            if (!isValidTrainingData()) {
                JOptionPane.showMessageDialog(this, "You need at least two folders in the 'images' directory to train classifiers.", "Training Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            String photoName = photoNameField.getText().trim();
            String baseDir = "images";

            if (photoName.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter the photo name to train.", "Input Required", JOptionPane.WARNING_MESSAGE);
                return;
            }

            JDialog loadingDialog = new JDialog(this, "Training", true);
            JLabel loadingLabel = new JLabel("Training... Please wait");
            loadingLabel.setHorizontalAlignment(JLabel.CENTER);
            loadingDialog.add(loadingLabel);
            loadingDialog.setSize(300, 100);
            loadingDialog.setLocationRelativeTo(this);
            SwingWorker<Void, Void> worker = new SwingWorker<>() {
                @Override
                protected Void doInBackground() {
                    try {
                        MultiPersonSVMTrainer trainer = new MultiPersonSVMTrainer();
                        trainer.trainAndSaveForUser(baseDir, photoName);
                    } catch (IOException ex) {
                        SwingUtilities.invokeLater(() -> {
                            JOptionPane.showMessageDialog(Main.this, "Failed to train classifiers: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                        });
                        ex.printStackTrace();
                    }
                    return null;
                }

                @Override
                protected void done() {
                    loadingDialog.dispose();
                    JOptionPane.showMessageDialog(Main.this, "Classifiers trained and saved successfully!", "Success", JOptionPane.INFORMATION_MESSAGE);
                }
            };

            worker.execute();
            loadingDialog.setVisible(true);
        });


        testButton.addActionListener(e -> {
            try {
                Map<String, SVM> userModels = ModelLoader.loadAllUserModels("images");

                if (userModels.size() < 2) {
                    JOptionPane.showMessageDialog(this, "You need at least two trained models to test the classifier.", "Test Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                SVMTester tester = new SVMTester();
                tester.loadModel("Model/trained_head_detector.model");
                new WebcamTester(userModels, tester);

            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Error opening webcam or loading models: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
                ex.printStackTrace();
            }
        });


        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void styleButton(JButton button) {
        button.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2));
        button.setFocusPainted(false);
        button.setBackground(new Color(70, 130, 180));
        button.setForeground(Color.WHITE);
        button.setFont(button.getFont().deriveFont(14f));
    }
    private boolean isValidTrainingData() {
        File imagesDir = new File("images");
        if (imagesDir.exists() && imagesDir.isDirectory()) {
            File[] subDirs = imagesDir.listFiles(File::isDirectory);
            return subDirs != null && subDirs.length >= 2;
        }
        return false;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::new);
    }
}
