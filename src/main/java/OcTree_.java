import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
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

public class OcTree_ implements PlugIn {

    private Node root; // Root node of the quadtree
    private int minSize; // Minimum block length for stopping division
    private float alpha; // Significance level
    private int imageWidth; // Width of the image (for pixel indexing)
    private int imageHeight; // Height of the image
    private int imageDepth; // Depth of the image
    private int d; // Number of dimensions in the data (e.g., 2 for 2D data)
    private int maxIterations; // Maximum iterations for M-estimator
    private float varAlpha; // Fraction of leaf nodes used to calculate variance (suggested 0.5 to 0.75)

    @Override
    public void run(String arg) {

        // ---- Display dialog box for user input ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Noise variance stabilisation 3D (Octree)");
        gd.addNumericField("Minimum leaf size: ", 4);
        gd.addNumericField("Alpha: ", 0.01f);
        gd.addNumericField("Max iterations (M-estimator): ", 50);
        gd.addNumericField("Fraction of leaf nodes (Least trimmed squares):", 0.75f);
        gd.showDialog();
        if (gd.wasCanceled()) return; // Stop the program if the "cancel" button is pressed

        IJ.log("Stabilising noise variance 3D (Octree method)...");

        // Quadtree parameters
        minSize = (int)gd.getNextNumber(); // Minimum length of the squares
        alpha = (float)gd.getNextNumber(); // Significance level for the F-distribution. Smaller values results in less stringency (larger regions).
        maxIterations = (int)gd.getNextNumber(); // Maximum number of iterations for M-estimator
        varAlpha = (float)gd.getNextNumber();
        d = 3;

        // Get the active image (or stop the program is none is found)
        ImagePlus image = ij.WindowManager.getCurrentImage();
        if (image == null) {
            ij.IJ.error("No image is open");
            return;
        }

        // Get image dimensions and pixel data
        ImageStack stack = image.getStack();
        imageWidth = stack.getWidth();
        imageHeight = stack.getHeight();
        imageDepth = stack.getSize();

        // Check if image has at least 3 slices, otherwise kill program
        if (imageDepth < 3) {
            IJ.error("Image must have at least 3 slices. Please try again.");
            return;
        }

        // Get image array
        float[] imageData = new float[imageWidth*imageHeight*imageDepth];
        for(int z=0; z<imageDepth; z++){
            FloatProcessor fp = stack.getProcessor(z+1).convertToFloatProcessor();
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    imageData[imageWidth*imageHeight*z+y*imageWidth+x] = fp.getf(y*imageWidth+x);
                }
            }
        }

        // Create and build the quadtree
        IJ.log("Building OcTree...");
        OcTree_ ocTree = new OcTree_(imageWidth, imageHeight, imageDepth, minSize, alpha);
        ocTree.buildTree(imageData);

        // Get robust mean and variance estimations from the QuadTree nodes
        IJ.log("Calculating GAT parameters...");
        ocTree.calculateRobustMeans(imageData, maxIterations);
        ocTree.calculateLTSVariances(imageData, 0.75f); // Using 75% of data for variance estimation

        // Collect (mean, variance) pairs
        List<double[]> meanVariancePairs = ocTree.collectMeanVariancePairs();

        // Perform linear regression and calculate g0 and eDC
        float[] results = performLinearRegression(meanVariancePairs);
        float g0 = results[0];
        float eDC = results[1];

        IJ.log("g0 = " + g0);
        IJ.log("eDC = " + eDC);

        // Create and add the overlay to the active image
        //Overlay overlay = ocTree.createOverlay();
        //image.setOverlay(overlay);

        // Show the updated image with the overlay
        //image.updateAndDraw();

        // Display variance-stabilised image
        float[] gat = applyGATtree(imageData, imageWidth*imageHeight*imageDepth, g0, eDC);

        ImageStack imsFinal = new ImageStack(imageWidth, imageHeight, imageDepth);
        for(int z=0; z<imageDepth; z++){
            float[] temp = new float[imageWidth*imageHeight];
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    temp[y*imageWidth+x] = gat[imageWidth*imageHeight*z+y*imageWidth+x];
                }
            }
            FloatProcessor fp = new FloatProcessor(imageWidth, imageHeight, temp);
            imsFinal.setProcessor(fp, z+1);
        }
        ImagePlus impFinal = new ImagePlus("Variance-stabilised image (Octree)", imsFinal);
        impFinal.show();
        IJ.log("Done!");
    }


    // ---------------------- //
    // ---- User methods ---- //
    // ---------------------- //

    // Default constructor required by ImageJ
    public OcTree_() {
    }

    // Constructor class for the QuadTree node
    private class Node {
        int x, y, z, width, height, depth; // Region bounds
        Node[] children = new Node[8]; // Children nodes
        boolean isLeaf; // True if this is a leaf node
        float robustMean; // Robust mean for the block
        float ltsVariance; // Variance estimated using the LTS

        // Constructor for a quadtree node
        public Node(int x, int y, int z, int width, int height, int depth) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.width = width;
            this.height = height;
            this.depth = depth;
            this.isLeaf = true; // Initially, a node is a leaf
            this.robustMean = 0.0f;
            this.ltsVariance = 0.0f;
        }
    }

    // Constructor for the quadtree
    public OcTree_(int imageWidth, int imageHeight, int imageDepth, int minSize, float alpha) {
        this.root = new Node(0, 0, 0, imageWidth, imageHeight, imageDepth);
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageDepth = imageDepth;
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

        if (node.width <= minSize || node.height <= minSize  || node.depth<= minSize || !divide) {
            node.isLeaf = true;
            return;
        }

        // Split the node into 4 quadrants
        splitNode(node);

        // Recursively process each child
        buildTreeRecursive(node.children[0], imageData);
        buildTreeRecursive(node.children[1], imageData);
        buildTreeRecursive(node.children[2], imageData);
        buildTreeRecursive(node.children[3], imageData);
        buildTreeRecursive(node.children[4], imageData);
        buildTreeRecursive(node.children[5], imageData);
        buildTreeRecursive(node.children[6], imageData);
        buildTreeRecursive(node.children[7], imageData);
    }

    // Split a node into 4 children
    private void splitNode(Node node) {
        int halfWidth = node.width / 2;
        int halfHeight = node.height / 2;
        int halfDepth = node.depth/2;

        node.children[0] = new Node(node.x, node.y, node.z, halfWidth, halfHeight, halfDepth);
        node.children[1] = new Node(node.x + halfWidth, node.y, node.z, halfWidth, halfHeight, halfDepth);
        node.children[2] = new Node(node.x, node.y + halfHeight, node.z, halfWidth, halfHeight, halfDepth);
        node.children[3] = new Node(node.x + halfWidth, node.y + halfHeight, node.z, halfWidth, halfHeight, halfDepth);
        node.children[4] = new Node(node.x, node.y, node.z + halfDepth, halfWidth, halfHeight, halfDepth);
        node.children[5] = new Node(node.x + halfWidth, node.y, node.z + halfDepth, halfWidth, halfHeight, halfDepth);
        node.children[6] = new Node(node.x, node.y + halfHeight, node.z + halfDepth, halfWidth, halfHeight, halfDepth);
        node.children[7] = new Node(node.x + halfWidth, node.y + halfHeight, node.z + halfDepth, halfWidth, halfHeight, halfDepth);

        node.isLeaf = false;
    }

    // Compute the splitting criterion (example: variance of pixel intensities)
    private boolean computeCriterion(Node node, float[] imageData) {

        // Calculate node variance
        float blockVariance = (float)Utils.getMeanAndVarBlock3D(imageData, imageWidth, imageHeight, node.x, node.y, node.z, node.x+node.width, node.y+node.height, node.z+node.depth)[1];

        // Calculate noise variance
        float noiseVariance = calculateNoiseVariance3D(node, imageData);

        // Calculate F-statistic
        boolean result = performFisherTest(blockVariance, noiseVariance, node.width*node.height, alpha);

        return result;
    }

    private float calculateNoiseVariance3D(Node node, float[] imageData) {
        int xStart = node.x;
        int yStart = node.y;
        int zStart = node.z;
        int xEnd = Math.min(node.x + node.width, imageWidth);
        int yEnd = Math.min(node.y + node.height, imageHeight);
        int zEnd = Math.min(node.z + node.depth, imageDepth);

        //int l = 2 * d + 1; // For 2D, l = 5 (2D neighborhood with center pixel)
        float l = 7.0f; // For 3D, d=3 thus l = 2*d+1 = 2*3+1 = 7
        float scaleFactor = (float) (1.0 / sqrt((double)(l * l + l))); // Scaling factor for pseudo-residuals

        // Compute pseudo-residuals
        List<Float> residuals = new ArrayList<>(); // List to store pseudo-residuals
        for(int z = zStart; z < zEnd; z++) {
            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    float v = imageData[imageWidth*imageHeight*z+y*imageWidth+x];

                    // Calculate Laplacian ∆Zi
                    float laplacian = l * v;
                    if (x > 0) laplacian -= imageData[imageWidth*imageHeight*z + y * imageWidth + (x - 1)];
                    if (x < imageWidth - 1) laplacian -= imageData[imageWidth*imageHeight*z + y * imageWidth + (x + 1)];
                    if (y > 0) laplacian -= imageData[imageWidth*imageHeight*z + (y - 1) * imageWidth + x];
                    if (y < imageHeight - 1) laplacian -= imageData[imageWidth*imageHeight*z + (y + 1) * imageWidth + x];
                    if (z > 0) laplacian -= imageData[imageWidth*imageHeight*(z-1) + y * imageWidth + x];
                    if (z < imageDepth - 1) laplacian -= imageData[imageWidth*imageHeight*(z+1) + y * imageWidth + x];

                    // Compute pseudo-residual
                    float residual = laplacian * scaleFactor;

                    // Add residual to residuals list
                    residuals.add(residual);
                }
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

    // Method to create an overlay for the quadtree NOT UPDATED
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
        int zStart = node.z;
        int xEnd = node.x+node.width;
        int yEnd = node.y + node.height;
        int zEnd = node.z + node.depth;

        // Initialize parameters
        float mean = 0.0f;
        float variance = 0.0f;
        int nPixels = 0;

        // Compute initial naive mean and variance
        for(int z = zStart; z< zEnd; z++) {
            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    float value = imageData[imageWidth*imageHeight*z + y * imageWidth + x];
                    mean += value;
                    variance += (value - mean) * (value - mean);
                    nPixels++;
                }
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

            for(int z = zStart; z< zEnd; z++) {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = xStart; x < xEnd; x++) {
                        float value = imageData[imageWidth*imageHeight*z + y * imageWidth + x];
                        float weight = (float) Math.exp(-((double) (value - mean) * (double) (value - mean)) / (2.0d * (double) variance));
                        weightSum += weight;
                        weightedSum += weight * value;
                    }
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
        int zStart = node.z;

        int xEnd = xStart + node.width;
        int yEnd = yStart + node.height;
        int zEnd = zStart + node.depth;

        // Collect squared residuals
        List<Float> residuals = new ArrayList<>();
        for(int z = zStart; z < zEnd; z++) {
            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    float v = imageData[imageWidth*imageHeight*z + y * imageWidth + x];
                    float residual = (v - robustMean) * (v - robustMean);
                    residuals.add(residual);
                }
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
            for(Node child : node.children){
                traverseTree(child, action);
            }
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
