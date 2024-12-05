import ij.IJ;
import ij.ImagePlus;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.distribution.FDistribution;

public class QuadTree_ implements PlugIn {

    private Node root; // Root node of the quadtree
    private int minSize; // Minimum block length for stopping division
    private float alpha; // Significance level
    private int imageWidth; // Width of the image (for pixel indexing)
    private int imageHeight; // Height of the image
    private int d; // Number of dimensions in the data (e.g., 2 for 2D data)

    @Override
    public void run(String arg) {
        // Get the active image
        ImagePlus image = ij.WindowManager.getCurrentImage();

        if (image == null) {
            ij.IJ.error("No image is open");
            return;
        }

        // Get image dimensions and pixel data
        imageWidth = image.getWidth();
        imageHeight = image.getHeight();
        FloatProcessor processor = image.getProcessor().convertToFloatProcessor();
        float[] imageData = (float[])processor.getPixels();

        // Quadtree parameters
        minSize = 4; // Minimum length of the squares
        alpha = 0.05f; // Significance level for the F-distribution
        d = 2;

        // Create and build the quadtree
        IJ.log("Building QuadTree...");
        QuadTree_ quadTree = new QuadTree_(imageWidth, imageHeight, minSize, alpha);
        quadTree.buildTree(imageData);

        // Get robust mean and variance estimations from the QuadTree nodes
        IJ.log("Calculating GAT parameters...");


        // Create and add the overlay to the active image
        Overlay overlay = quadTree.createOverlay();
        image.setOverlay(overlay);

        // Show the updated image with the overlay
        image.updateAndDraw();
    }


    // ---------------------- //
    // ---- User methods ---- //
    // ---------------------- //

    // Default constructor required by ImageJ
    public QuadTree_() {
    }


    // Constructor class for the QuadTree node
    private class Node {
        int x, y, width, height; // Region bounds
        Node topLeft, topRight, bottomLeft, bottomRight; // Children nodes
        boolean isLeaf; // True if this is a leaf node

        // Constructor for a quadtree node
        public Node(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.isLeaf = true; // Initially, a node is a leaf
        }
    }


    // Constructor for the quadtree
    public QuadTree_(int imageWidth, int imageHeight, int minSize, float alpha) {
        this.root = new Node(0, 0, imageWidth, imageHeight);
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.minSize = minSize;
        this.alpha = alpha;
    }

    // Method to build the quadtree
    public void buildTree(float[] imageData) {
        buildTreeRecursive(root, imageData);
    }

    // Recursive function to build the quadtree
    private void buildTreeRecursive(Node node, float[] imageData) {
        // Check if we should stop splitting
        boolean divide = computeCriterion(node, imageData);

        if (node.width <= minSize || node.height <= minSize || !divide) {
            node.isLeaf = true;
            return;
        }

        // Split the node into 4 quadrants
        splitNode(node);

        // Recursively process each child
        buildTreeRecursive(node.topLeft, imageData);
        buildTreeRecursive(node.topRight, imageData);
        buildTreeRecursive(node.bottomLeft, imageData);
        buildTreeRecursive(node.bottomRight, imageData);
    }

    // Split a node into 4 children
    private void splitNode(Node node) {
        int halfWidth = node.width / 2;
        int halfHeight = node.height / 2;

        node.topLeft = new Node(node.x, node.y, halfWidth, halfHeight);
        node.topRight = new Node(node.x + halfWidth, node.y, halfWidth, halfHeight);
        node.bottomLeft = new Node(node.x, node.y + halfHeight, halfWidth, halfHeight);
        node.bottomRight = new Node(node.x + halfWidth, node.y + halfHeight, halfWidth, halfHeight);

        node.isLeaf = false; // This node is no longer a leaf
    }

    // Compute the splitting criterion (example: variance of pixel intensities)
    private boolean computeCriterion(Node node, float[] imageData) {

        // Calculate node variance
        float blockVariance = Utils.getMeanAndVarBlock2D(imageData, imageWidth, node.x, node.y, node.x+node.width, node.y+node.height)[1];

        // Calculate noise variance
        float noiseVariance = calculateNoiseVariance(node, imageData);

        // Calculate F-statistic
        boolean result = performFisherTest(blockVariance, noiseVariance, node.width*node.height, alpha);

        return result;
    }

    private float calculateNoiseVariance(Node node, float[] imageData) {
        int xStart = node.x;
        int yStart = node.y;
        int xEnd = Math.min(node.x + node.width, imageWidth);
        int yEnd = Math.min(node.y + node.height, imageHeight);

        //int l = 2 * d + 1; // For 2D, l = 5 (2D neighborhood with center pixel)
        float l = 5.0f;
        float scaleFactor = (float) (1.0 / Math.sqrt((double)(l * l + l))); // Scaling factor for pseudo-residuals

        // Compute pseudo-residuals
        List<Float> residuals = new ArrayList<>(); // List to store pseudo-residuals
        for (int y = yStart; y < yEnd; y++) {
            for (int x = xStart; x < xEnd; x++) {
                float z = imageData[y * imageWidth + x];

                // Calculate Laplacian ∆Zi
                float laplacian = l * z;
                if (x > 0) laplacian -= imageData[y * imageWidth + (x - 1)]; // Left neighbor
                if (x < imageWidth - 1) laplacian -= imageData[y * imageWidth + (x + 1)]; // Right neighbor
                if (y > 0) laplacian -= imageData[(y - 1) * imageWidth + x]; // Top neighbor
                if (y < imageHeight - 1) laplacian -= imageData[(y + 1) * imageWidth + x]; // Bottom neighbor

                // Compute pseudo-residual
                float residual = laplacian * scaleFactor;

                // Add residual to residuals list
                residuals.add(residual);
            }
        }

        // Compute mean of residuals
        float sumResiduals = 0.0f;
        for (float residual : residuals) {
            sumResiduals += residual;
        }
        float meanResidual = sumResiduals / (float)residuals.size();

        // Compute variance of residuals
        float sumSquaredDifferences = 0.0f;
        for (float residual : residuals) {
            sumSquaredDifferences += (float)Math.pow((double)residual - (double)meanResidual, 2.0);
        }
        float variance = sumSquaredDifferences / ((float)residuals.size() - 1.0f);

        return variance;
    }


    public static boolean performFisherTest(float varianceZ, float varianceE, int regionSize, float alpha) {
        // Compute the F-statistic
        float smallerVariance = Math.min(varianceZ, varianceE);
        float largerVariance = Math.max(varianceZ, varianceE);
        float ratio = smallerVariance / largerVariance;

        // Calculate the F-distribution threshold T_alphaF with (regionSize - 1) degrees of freedom
        int degreesOfFreedom = regionSize - 1;
        FDistribution fDistribution = new FDistribution(degreesOfFreedom, degreesOfFreedom);
        float threshold = (float) fDistribution.inverseCumulativeProbability((double)alpha);

        // Compare the F-statistic to the threshold
        return ratio < threshold; // If ratio (F-statistic) is smaller than threshold, variance is significantly different than noise variance and we should divide more
    }

    // Method to create an overlay for the quadtree
    public Overlay createOverlay() {
        Overlay overlay = new Overlay();
        addNodeToOverlay(root, overlay);
        return overlay;
    }

    // Recursive helper method to add nodes to the overlay
    private void addNodeToOverlay(Node node, Overlay overlay) {
        if (node.isLeaf) {
            // Create a rectangle for the leaf node
            Roi rectangle = new Roi(node.x, node.y, node.width, node.height);
            rectangle.setStrokeColor(java.awt.Color.RED); // Set rectangle color
            rectangle.setStrokeWidth(1); // Set rectangle stroke width
            overlay.add(rectangle);
        } else {
            // Recursively process children
            addNodeToOverlay(node.topLeft, overlay);
            addNodeToOverlay(node.topRight, overlay);
            addNodeToOverlay(node.bottomLeft, overlay);
            addNodeToOverlay(node.bottomRight, overlay);
        }
    }

    // Calculate robust mean of a QuadTree Node using a Leclerc influence function
    //private float calculateRobusMean(Node node, float[] imageData, float sigma)

}
