import ij.IJ;
import ij.ImagePlus;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import static java.lang.Math.sqrt;

public class QuadTree_ implements PlugIn {

    private Node root; // Root node of the quadtree
    private int minSize; // Minimum block length for stopping division
    private float alpha; // Significance level
    private int imageWidth; // Width of the image (for pixel indexing)
    private int imageHeight; // Height of the image
    private int d; // Number of dimensions in the data (e.g., 2 for 2D data)
    private int maxIterations; // Maximum iterations for robust means calculations
    private float varAlpha; // Fraction of leaf nodes used to calculate variance (suggested 0.5 to 0.75)

    @Override
    public void run(String arg) {

        // ---- Display dialog box for user input ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Noise variance stabilisation 2D (Quadtree)");
        gd.addNumericField("Minimum leaf size: ", 4);
        gd.addNumericField("Alpha: ", 0.01f);
        gd.addNumericField("Max iterations (M-estimator): ", 50);
        gd.addNumericField("Fraction of leaf nodes (Least trimmed squares):", 0.75f);
        gd.showDialog();
        if (gd.wasCanceled()) return; // Stop the program if the "cancel" button is pressed

        IJ.log("Stabilising noise variance 3D (Quadtree method)...");

        // Quadtree parameters
        minSize = (int)gd.getNextNumber(); // Minimum length of the squares
        alpha = (float)gd.getNextNumber(); // Significance level for the F-distribution. Smaller values results less stringency (larger regions).
        maxIterations = (int)gd.getNextNumber(); // Maximum umber of iterations in M-estimator
        varAlpha = (float)gd.getNextNumber(); // Fraction of leaf nodes used for LTS variance calculation
        d = 2; // Number of dimensions

        // Get the active image (or stop the program is none is found)
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

        // Create and build the quadtree
        IJ.log("Building QuadTree...");
        QuadTree_ quadTree = new QuadTree_(imageWidth, imageHeight, minSize, alpha);
        quadTree.buildTree(imageData);

        // Get robust mean and variance estimations from the QuadTree nodes
        IJ.log("Calculating GAT parameters...");
        quadTree.calculateRobustMeans(imageData, maxIterations);
        quadTree.calculateLTSVariances(imageData, varAlpha);

        // Collect (mean, variance) pairs
        List<double[]> meanVariancePairs = quadTree.collectMeanVariancePairs();

        // Perform linear regression and calculate g0 and eDC
        float[] results = performLinearRegression(meanVariancePairs);
        float g0 = results[0];
        float eDC = results[1];

        IJ.log("g0: " + g0);
        IJ.log("eDC: " + eDC);

        // Create and add the overlay to the active image
        Overlay overlay = quadTree.createOverlay();
        image.setOverlay(overlay);

        // Show the updated image with the overlay
        image.updateAndDraw();

        // SHOW GAT
        float[] gat = applyGATtree(imageData, imageWidth*imageHeight, g0, eDC);
        FloatProcessor ipFinal = new FloatProcessor(imageWidth, imageHeight, gat);
        ImagePlus impFinal = new ImagePlus("Variance-stabilised image (Quadtree)", ipFinal);
        impFinal.show();
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
        float robustMean; // Robust mean for the block
        float ltsVariance; // Variance estimated using the LTS

        // Constructor for a quadtree node
        public Node(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.isLeaf = true; // Initially, a node is a leaf
            this.robustMean = 0.0f;
            this.ltsVariance = 0.0f;
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
        float scaleFactor = (float) (1.0 / sqrt((double)(l * l + l))); // Scaling factor for pseudo-residuals

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
        traverseTree(root, node -> {
            if (node.isLeaf) {
                Roi rectangle = new Roi(node.x, node.y, node.width, node.height);
                rectangle.setStrokeColor(java.awt.Color.RED);
                rectangle.setStrokeWidth(1);
                overlay.add(rectangle);
            }
        });
        return overlay;
    }


    // Calculate robust mean of a QuadTree Node using a Leclerc influence function
    private float calculateRobustMean(Node node, float[] imageData, int maxIterations) {
        int xStart = node.x;
        int yStart = node.y;
        int xEnd = node.x+node.width;
        int yEnd = node.y+ node.height;

        // Initialize parameters
        float mean = 0.0f;
        float variance = 0.0f;
        int nPixels = 0;

        // Compute initial naive mean and variance
        for (int y=yStart; y<yEnd; y++) {
            for (int x=xStart; x<xEnd; x++) {
                float value = imageData[y*imageWidth+x];
                mean += value;
                variance += (value - mean) * (value - mean);
                nPixels++;
            }
        }
        mean /= (float)nPixels;
        variance /= (float)(nPixels - 1);
        variance = Math.max(variance, Utils.EPSILON); // Ensure variance is non-zero to avoid the exponential term below blowing up to infinity or become undefined

        // Iteratively compute the robust mean
        float prevMean;
        int iteration = 0;

        do {
            prevMean = mean;
            float weightSum = 0.0f;
            float weightedSum = 0.0f;

            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    float value = imageData[y * imageWidth + x];
                    float weight = (float) Math.exp(-((double)(value - mean) * (double)(value - mean)) / (2.0d * (double)variance));
                    weightSum += weight;
                    weightedSum += weight * value;
                }
            }

            // Update mean
            mean = weightedSum / Math.max(weightSum, Utils.EPSILON); // Ensure weightSum is nonzero to avoid division by zero
            iteration++;
        } while (Math.abs(mean - prevMean) > 1e-5 && iteration < maxIterations);

        return mean;
    }


    public void calculateRobustMeans(float[] imageData, int maxIterations) {
        traverseTree(root, node -> {
            if (node.isLeaf) {
                node.robustMean = calculateRobustMean(node, imageData, maxIterations);
            }
        });
    }



    private float calculateLTSVariance(Node node, float[] imageData, float robustMean, float alpha) {
        int xStart = node.x;
        int yStart = node.y;
        int xEnd = xStart + node.width;
        int yEnd = yStart + node.height;

        // Collect squared residuals
        List<Float> residuals = new ArrayList<>();
        for (int y = yStart; y < yEnd; y++) {
            for (int x = xStart; x < xEnd; x++) {
                float z = imageData[y * imageWidth + x];
                float residual = (z - robustMean) * (z - robustMean);
                residuals.add(residual);
            }
        }

        // Sort residuals
        residuals.sort(Float::compare);

        // Compute the number of residuals to trim based on alpha
        int n = residuals.size();
        int h = (int) Math.floor(alpha * (float)n); // Retain alpha fraction of residuals

        // Sum the smallest "h" residuals
        float trimmedSum = 0.0f;
        for (int i = 0; i < h; i++) {
            trimmedSum += residuals.get(i);
        }

        // Return the trimmed variance
        return trimmedSum / ((float)h+Utils.EPSILON);
    }

    public void calculateLTSVariances(float[] imageData, float alpha) {
        traverseTree(root, node -> {
            if (node.isLeaf) {
                node.ltsVariance = calculateLTSVariance(node, imageData, node.robustMean, alpha);
            }
        });
    }

    // Traverse tree
    private void traverseTree(Node node, Consumer<Node> action) {
        if (node == null) return;

        // Perform the action on the current node
        action.accept(node);

        // Recursively traverse children if they exist
        if (!node.isLeaf) {
            traverseTree(node.topLeft, action);
            traverseTree(node.topRight, action);
            traverseTree(node.bottomLeft, action);
            traverseTree(node.bottomRight, action);
        }
    }

    // Collect mean and variance pairs from QuadTree
    public List<double[]> collectMeanVariancePairs(){
        List<double[]> pairs = new ArrayList<>();
        traverseTree(root, node -> {
            if(node.isLeaf){
                pairs.add(new double[]{(double) node.robustMean, (double) node.ltsVariance});
            }
        });

        return pairs;
    }


    // Do linear regression
    public float[] performLinearRegression(List<double[]> pairs){
        SimpleRegression regression = new SimpleRegression();
        for(double[] pair : pairs){
            regression.addData(pair[0], pair[1]);
        }
        regression.regress();

        float g0 = (float)regression.getSlope();
        float eDC = (float)regression.getIntercept();

        return new float[]{g0, eDC};
    }

    public float[] applyGATtree(float[] imageData, int nPixels, float g0, float eDC){

        float[] gat = new float[nPixels];
        double refConstant = 3.0d/8.0d * (double)g0 * (double)g0 + (double)eDC; //threshold to avoid taking the square root of negative values.
        for(int i =0; i<nPixels; i++) {
            double v = (double)imageData[i];
            if(v<-refConstant/g0){
                v = 0.0d;
            }else{
                v = ((2.0d/(double)g0) * sqrt((double)g0 * v + refConstant));
            }
            gat[i] = (float) v;
        }
        return gat;
    }
}
