package Camera;

import java.awt.BorderLayout;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingConstants;
import javax.swing.SwingWorker;
import javax.swing.Timer;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import Hog.HogExtractor;
import Utils.SVMTester;

public class CameraCaptureGUI extends JFrame {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private VideoCapture capture;
    private JLabel imageLabel;
    private String photoName;
    private int numberOfPhotos;
    private int photosTaken = 0;
    private int startIndex = 0;
    private Timer timer;
    private boolean isShooting = false;

    private JButton startShootingButton;
    private JTextArea statusTextArea;

    private SVMTester svmTester;
    private HogExtractor hogExtractor;

    public CameraCaptureGUI(String photoName, int numberOfPhotos) {
        super("Camera Capture");
        this.photoName = photoName;
        this.numberOfPhotos = numberOfPhotos;

        svmTester = new SVMTester();
        hogExtractor = new HogExtractor();
        svmTester.loadModel("Model/trained_head_detector.model");

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(1100, 900);
        setLayout(new BorderLayout());

        JPanel cameraPanel = new JPanel(new BorderLayout());
        cameraPanel.setBorder(BorderFactory.createTitledBorder("Camera Feed"));

        imageLabel = new JLabel("", SwingConstants.CENTER);
        cameraPanel.add(imageLabel, BorderLayout.CENTER);

        startShootingButton = new JButton("Start Shooting");
        startShootingButton.addActionListener(e -> {
            if (!isShooting) {
                clearRawImagesDirectory();
                photosTaken = 0;
                startIndex = getNextAvailableIndex(photoName);
                isShooting = true;
                startShootingButton.setText("Stop Shooting");
                statusTextArea.setText("Starting capture...\n");
            } else {
                isShooting = false;
                startShootingButton.setText("Start Shooting");
                statusTextArea.append("Capture stopped.\n");
            }
        });

        JPanel bottomPanel = new JPanel(new BorderLayout());
        bottomPanel.add(startShootingButton, BorderLayout.NORTH);

        statusTextArea = new JTextArea(5, 30);
        statusTextArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(statusTextArea);
        bottomPanel.add(scrollPane, BorderLayout.CENTER);

        add(cameraPanel, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);

        capture = new VideoCapture(0);
        capture.set(3, 320);
        capture.set(4, 240);
        if (!capture.isOpened()) {
            JOptionPane.showMessageDialog(this, "Cannot open webcam", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        setLocationRelativeTo(null);
        setVisible(true);

        startCapture();
    }

    public void startCapture() {
        timer = new Timer(100, e -> {
            Mat frame = new Mat();
            if (capture.read(frame)) {
                BufferedImage fullImage = matToBufferedImage(frame);

                if (isShooting && photosTaken < numberOfPhotos) {
                    saveFullFrame(fullImage);
                    statusTextArea.append("Captured photo " + (photosTaken) + " of " + numberOfPhotos + "\n");

                    if (photosTaken >= numberOfPhotos) {
                        isShooting = false;
                        startShootingButton.setText("Start Shooting");
                        JOptionPane.showMessageDialog(this, "Finished capturing " + numberOfPhotos + " photos.");

                        JDialog loadingDialog = new JDialog(this, "Processing", true);
                        JLabel loadingLabel = new JLabel("Processing captured images... Please wait.", SwingConstants.CENTER);
                        loadingDialog.add(loadingLabel);
                        loadingDialog.setSize(300, 100);
                        loadingDialog.setLocationRelativeTo(this);

                        SwingWorker<Void, Void> worker = new SwingWorker<>() {
                            @Override
                            protected Void doInBackground() {
                                processCapturedImages();
                                return null;
                            }

                            @Override
                            protected void done() {
                                loadingDialog.dispose();
                            }
                        };

                        worker.execute();
                        loadingDialog.setVisible(true);
                    }
                }

                BufferedImage displayImage = resizeImage(fullImage, 620, 480);
                imageLabel.setIcon(new ImageIcon(displayImage));
            }
        });
        timer.start();
    }

    private void saveFullFrame(BufferedImage frameImage) {
        if (photosTaken < numberOfPhotos) {
            try {
                File dir = new File("raw_training_data");
                if (!dir.exists()) {
                    dir.mkdirs();
                }
                String filename = photoName + "_" + (startIndex + photosTaken) + ".png";
                ImageIO.write(frameImage, "png", new File(dir, filename));
                photosTaken++;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private int getNextAvailableIndex(String photoName) {
        File outputDir = new File("images" + File.separator + photoName);
        if (!outputDir.exists()) {
            return 0;
        }

        int maxIndex = -1;
        Pattern pattern = Pattern.compile(Pattern.quote(photoName) + "_(\\d+)\\.png");

        File[] files = outputDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
        if (files != null) {
            for (File f : files) {
                Matcher m = pattern.matcher(f.getName());
                if (m.matches()) {
                    try {
                        int index = Integer.parseInt(m.group(1));
                        if (index > maxIndex) {
                            maxIndex = index;
                        }
                    } catch (NumberFormatException ignored) {}
                }
            }
        }

        return maxIndex + 1;
    }

    private void processCapturedImages() {
        File rawDir = new File("raw_training_data");
        File[] files = rawDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) {
            JOptionPane.showMessageDialog(this, "No raw images found to process.", "Processing", JOptionPane.INFORMATION_MESSAGE);
            return;
        }

        svmTester = new SVMTester();
        hogExtractor = new HogExtractor();
        svmTester.loadModel("Model/trained_head_detector.model");

        File outputDir = new File("images" + File.separator + photoName);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        int processedCount = 0;
        for (File file : files) {
            try {
                BufferedImage img = ImageIO.read(file);
                var detections = svmTester.getDetections(img);

                if (!detections.isEmpty()) {
                    var largestRect = detections.get(0);
                    int maxArea = largestRect.width * largestRect.height;

                    for (var rect : detections) {
                        int area = rect.width * rect.height;
                        if (area > maxArea) {
                            maxArea = area;
                            largestRect = rect;
                        }
                    }

                    BufferedImage headImg = img.getSubimage(largestRect.x, largestRect.y, largestRect.width, largestRect.height);
                    BufferedImage resizedHead = resizeImage(headImg, 128, 128);

                    String outName = file.getName();
                    ImageIO.write(resizedHead, "png", new File(outputDir, outName));
                    processedCount++;
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        JOptionPane.showMessageDialog(this, "Processed " + processedCount + " images for training.", "Processing Complete", JOptionPane.INFORMATION_MESSAGE);
    }

    private void clearRawImagesDirectory() {
        File rawDir = new File("raw_training_data");
        if (rawDir.exists()) {
            File[] files = rawDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    f.delete();
                }
            }
        }
    }

    private BufferedImage resizeImage(BufferedImage image, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        byte[] b = new byte[mat.channels() * mat.cols() * mat.rows()];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }
}
