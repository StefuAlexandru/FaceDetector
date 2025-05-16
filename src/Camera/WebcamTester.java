package Camera;

import Hog.HogExtractor;
import Utils.SVMTester;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import javax.swing.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class WebcamTester extends JFrame {

    private final VideoCapture capture;
    private final JLabel imageLabel;
    private final javax.swing.Timer timer;
    private final Map<String, SVM> personModels;
    private final SVMTester faceDetector;
    private final HogExtractor hogExtractor = new HogExtractor();

    private static final int FPS = 10;
    private static final int WINDOW_SIZE = 128;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public WebcamTester(Map<String, SVM> personModels, SVMTester faceDetector) throws IOException {
        super("Live Face Recognition");

        this.personModels = personModels;
        this.faceDetector = faceDetector;

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLayout(new BorderLayout());

        imageLabel = new JLabel();
        add(imageLabel, BorderLayout.CENTER);

        capture = new VideoCapture(0);

        if (!capture.isOpened()) {
            throw new IOException("Cannot open webcam");
        }

        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 320);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 240);

        setLocationRelativeTo(null);
        setVisible(true);
        imageLabel.setHorizontalAlignment(SwingConstants.CENTER);
        imageLabel.setVerticalAlignment(SwingConstants.CENTER);

        int delay = 1000 / FPS;
        timer = new javax.swing.Timer(delay, e -> {
            Mat frame = new Mat();
            if (capture.read(frame)) {
                Mat resizedFrame = new Mat();
                Imgproc.resize(frame, resizedFrame, new Size(320, 240));
                BufferedImage image = matToBufferedImage(resizedFrame);
                BufferedImage processed = processFrame(image);
                imageLabel.setIcon(new ImageIcon(processed));
                imageLabel.setPreferredSize(new Dimension(640, 480));
                imageLabel.revalidate();
            }
        });

        timer.start();
    }

    private BufferedImage processFrame(BufferedImage frame) {
        BufferedImage annotated = new BufferedImage(frame.getWidth(), frame.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g2d = annotated.createGraphics();
        g2d.drawImage(frame, 0, 0, null);
        g2d.setStroke(new BasicStroke(2));

        List<Rectangle> faces = faceDetector.getDetections(frame);

        for (Rectangle rect : faces) {
            try {
                BufferedImage cropped = frame.getSubimage(rect.x, rect.y, rect.width, rect.height);
                BufferedImage resized = resizeImage(cropped, WINDOW_SIZE, WINDOW_SIZE);
                double[] hog = hogExtractor.extractHOG(resized);

                Mat sampleMat = new Mat(1, hog.length, CvType.CV_32F);
                for (int i = 0; i < hog.length; i++) {
                    sampleMat.put(0, i, hog[i]);
                }

                String matchedUser = null;
                for (Map.Entry<String, SVM> entry : personModels.entrySet()) {
                    String user = entry.getKey();
                    SVM model = entry.getValue();
                    float prediction = model.predict(sampleMat);
                    if (prediction == 1.0f) {
                        matchedUser = user;
                        break;
                    }
                }

                g2d.setColor(Color.GREEN);
                g2d.drawRect(rect.x, rect.y, rect.width, rect.height);

                String label = (matchedUser != null) ? matchedUser : "Unknown";
                g2d.setColor((matchedUser != null) ? Color.RED : Color.GRAY);
                g2d.drawString(label, rect.x, rect.y - 5);

            } catch (Exception e) {
                // prevenim crash la subimage
                System.err.println("Eroare la procesarea unei regiuni: " + e.getMessage());
            }
        }

        g2d.dispose();
        return annotated;
    }

    private BufferedImage resizeImage(BufferedImage img, int width, int height) {
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }

    public void stop() {
        if (timer != null) {
            timer.stop();
        }
        if (capture != null) {
            capture.release();
        }
        dispose();
    }
}
