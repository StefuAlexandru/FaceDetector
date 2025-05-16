package Utils;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Image;
import java.io.File;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

public class ImageManagerGUI extends JFrame {

    private File imagesDir;
    private List<File> imageFiles;
    private int currentIndex = 0;

    private JLabel imageLabel;
    private JLabel imageNameLabel;
    private JButton prevButton;
    private JButton nextButton;
    private JButton deleteButton;

    public ImageManagerGUI(String imagesFolderPath) {
        super("Image Viewer");
        this.imagesDir = new File(imagesFolderPath);

        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setSize(400, 300);
        setLayout(new BorderLayout());

        imageLabel = new JLabel("", SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(128, 128));

        imageNameLabel = new JLabel("", SwingConstants.CENTER);

        JPanel imagePanel = new JPanel();
        imagePanel.setLayout(new BorderLayout());
        imagePanel.add(imageLabel, BorderLayout.CENTER);
        imagePanel.add(imageNameLabel, BorderLayout.SOUTH);
        add(imagePanel, BorderLayout.CENTER);

        JPanel controlsPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        prevButton = new JButton("←");
        nextButton = new JButton("→");
        deleteButton = new JButton("Delete Image");

        controlsPanel.add(prevButton);
        controlsPanel.add(deleteButton);
        controlsPanel.add(nextButton);
        add(controlsPanel, BorderLayout.SOUTH);

        prevButton.addActionListener(e -> showPreviousImage());
        nextButton.addActionListener(e -> showNextImage());
        deleteButton.addActionListener(e -> deleteCurrentImage());

        loadImages();

        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void loadImages() {
        File[] files = imagesDir.listFiles((dir, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg");
        });

        if (files == null || files.length == 0) {
            imageFiles = List.of();
            currentIndex = 0;
            imageLabel.setIcon(null);
            imageNameLabel.setText("No images available.");
        } else {
            imageFiles = Arrays.asList(files);
            currentIndex = Math.min(currentIndex, imageFiles.size() - 1);
            displayImage();
        }
    }

    private void displayImage() {
        if (imageFiles == null || imageFiles.isEmpty()) {
            imageLabel.setIcon(null);
            imageNameLabel.setText("No images to display.");
            return;
        }

        try {
            File imgFile = imageFiles.get(currentIndex);
            Image img = ImageIO.read(imgFile);
            if (img != null) {
                Image scaledImg = img.getScaledInstance(128, 128, Image.SCALE_SMOOTH);
                imageLabel.setIcon(new ImageIcon(scaledImg));
                imageNameLabel.setText(imgFile.getName());
            } else {
                imageLabel.setIcon(null);
                imageNameLabel.setText("Could not load image.");
            }
        } catch (Exception e) {
            e.printStackTrace();
            imageLabel.setIcon(null);
            imageNameLabel.setText("Error loading image.");
        }
    }

    private void showPreviousImage() {
        if (imageFiles == null || imageFiles.isEmpty()) return;
        currentIndex = (currentIndex - 1 + imageFiles.size()) % imageFiles.size();
        displayImage();
    }

    private void showNextImage() {
        if (imageFiles == null || imageFiles.isEmpty()) return;
        currentIndex = (currentIndex + 1) % imageFiles.size();
        displayImage();
    }

    private void deleteCurrentImage() {
        if (imageFiles == null || imageFiles.isEmpty()) return;

        File toDelete = imageFiles.get(currentIndex);
        int confirm = JOptionPane.showConfirmDialog(this,
                "Are you sure you want to delete this image?\n" + toDelete.getName(),
                "Delete Image", JOptionPane.YES_NO_OPTION);

        if (confirm == JOptionPane.YES_OPTION && toDelete.delete()) {
            loadImages();
        } else {
            JOptionPane.showMessageDialog(this, "Failed to delete image.", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }
}
