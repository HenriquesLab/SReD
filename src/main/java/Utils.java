import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.measure.Calibration;
import ij.plugin.LutLoader;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.LUT;
import java.awt.image.IndexColorModel;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.*;

public class Utils {

    public static float EPSILON = 0.0000001f;

    // Private constructor to prevent instantiation
    private Utils(){
        throw new UnsupportedOperationException("Utility class");
    }


    // -------------------------- //
    // ---- UNSORTED OBJECTS ---- //
    // -------------------------- //

    /**
     * Custom object to hold a relevance mask.
     */
    public static class RelevanceMask {
        private final float[] relevanceMask;
        private final float relevanceConstant;
        private final float threshold;
        private final float noiseMeanVariance;


        /**
         * Constructs a {@link RelevanceMask} object.
         *
         * @param relevanceMask         A flattened array of float values representing the binary relevance mask (values are either 0.0 or 1.0)
         * @param threshold             The relevance threshold used to calculate the relevance mask.
         * @param relevanceConstant     The relevance constant used to calculate the relevance mask.
         * @param noiseMeanVariance     The mean variance of the noise in the input image used to calculate the relevance mask.
         */
        public RelevanceMask(float[] relevanceMask, float relevanceConstant, float threshold, float noiseMeanVariance) {
            this.relevanceMask = relevanceMask;
            this.threshold = threshold;
            this.relevanceConstant = relevanceConstant;
            this.noiseMeanVariance = noiseMeanVariance;
        }


        /**
         * Returns the binary relevance mask as flattened 1D array.
         *
         * @return A flattened 1D array of float values containing the relevance mask.
         */
        public float[] getRelevanceMask() {
            return relevanceMask;
        }


        /**
         * Returns the relevance constant used to calculate the relevance mask.
         *
         * @return The relevance constant used to calculate the relevance mask.
         */
        public float getRelevanceConstant() {
            return relevanceConstant;
        }


        /**
         * Returns the relevance threshold used to calculate the relevance mask (relevance constant * noiseMeanVariance).
         *
         * @return The relevance constant used to calculate the relevance mask.
         */
        public float getRelevanceThreshold() {
            return threshold;
        }


        /**
         * Returns the relevance threshold used to calculate the relevance mask (relevance constant * noiseMeanVariance).
         *
         * @return The relevance constant used to calculate the relevance mask.
         */
        public float getNoiseMeanVariance() {
            return noiseMeanVariance;
        }
    }


    // ----------------------------------------- //
    // ---- OBJECTS FOR BLOCK REPETITION 2D ---- //
    // ----------------------------------------- //

    /**
     * This class represents a 2D reference block after processing an input square or rectangle.
     * Processing involves discarding pixels outside the inbound ellipse and normalizing the range.
     * It provides accessor methods to retrieve block dimensions and statistical information.
     */
    public static class ReferenceBlock2D {
        private final float[] pixels;
        private final int width;
        private final int height;
        private final int radiusWidth;
        private final int radiusHeight;
        private final int size;
        private final float mean;
        private final float std;

        /**
         * Constructs a ReferenceBlock2D object with the specified parameters.
         *
         * @param pixels A 1D array of float values containing the normalized block pixels.
         * @param width Block width in pixels.
         * @param height Block height in pixels.
         * @param radiusWidth Block radius along its width in pixels.
         * @param radiusHeight Block radius along its height in pixels.
         * @param size Total number of pixels in the block.
         * @param mean Mean value of the block pixels.
         * @param std Standard deviation of the block pixels.
         */
        public ReferenceBlock2D(float[] pixels, int width, int height, int radiusWidth,
                              int radiusHeight, int size, float mean, float std){

            this.pixels = pixels;
            this.width = width;
            this.height = height;
            this.radiusWidth = radiusWidth;
            this.radiusHeight = radiusHeight;
            this.size = size;
            this.mean = mean;
            this.std = std;
        }


        /**
         * Returns the normalized block pixels.
         *
         * @return A 1D array of float values containing the normalized block pixels retained
         *         after processing.
         */
        public float[] getPixels() {
            return pixels;
        }

        /**
         * Returns the block width in pixels.
         *
         * @return Block width.
         */
        public int getWidth() {
            return width;
        }

        /**
         * Returns the block height in pixels.
         *
         * @return Block height.
         */
        public int getHeight() {
            return height;
        }

        /**
         * Returns the block radius along its width in pixels.
         *
         * @return Block width radius.
         */
        public int getRadiusWidth() {
            return radiusWidth;
        }

        /**
         * Returns the block radius along its height in pixels.
         *
         * @return Block height radius.
         */
        public int getRadiusHeight() {
            return radiusHeight;
        }

        /**
         * Returns the total number of pixels in the block.
         *
         * @return Size of the block.
         */
        public int getSize() {
            return size;
        }

        /**
         * Returns the mean value of the block pixels.
         *
         * @return Mean of the block.
         */
        public float getMean() {
            return mean;
        }

        /**
         * Returns the standard deviation of the block pixels.
         *
         * @return Standard deviation of the block.
         */
        public float getStd() {
            return std;
        }
    }


    /**
     * Custom object to hold a 2D input image along with its dimensions and statistics.
     */
    public static class InputImage2D
    {
        private final float[] imageArray;
        private final int width;
        private final int height;
        private final int size;

        /**
         * Constructs an InputImage2D object with the specified image data and dimensions.
         *
         * @param imageArray A 1D array of float values representing the input image pixels.
         * @param width The width of the image in pixels.
         * @param height The height of the image in pixels.
         * @param size The total number of pixels in the image.
         */
        public InputImage2D(float[] imageArray, int width, int height, int size) {
            this.imageArray = imageArray;
            this.width = width;
            this.height = height;
            this.size = size;
        }

        /**
         * Returns the input image pixels as a 1D array.
         *
         * @return A 1D array of float values containing the input image pixels.
         */
        public float[] getImageArray() {
            return imageArray;
        }

        /**
         * Returns the width of the image in pixels.
         *
         * @return The width of the image.
         */
        public int getWidth() {
            return width;
        }

        /**
         * Returns the height of the image in pixels.
         *
         * @return The height of the image.
         */
        public int getHeight() {
            return height;
        }

        /**
         * Returns the total number of pixels in the image.
         *
         * @return The size of the image.
         */
        public int getSize() {
            return size;
        }
    }


    // ----------------------------------------- //
    // ---- OBJECTS FOR BLOCK REPETITION 3D ---- //
    // ----------------------------------------- //

    /**
     * This class represents a 3D reference block after processing an input square or rectangle.
     * Processing involves discarding pixels outside the inbound spheroid and normalizing the range.
     * It provides accessor methods to retrieve block dimensions and statistical information.
     */
    public static class ReferenceBlock3D
    {
        private final float[] pixels;
        private final int width;
        private final int height;
        private final int depth;
        private final int radiusWidth;
        private final int radiusHeight;
        private final int radiusDepth;
        private final int size;
        private final float mean;
        private final float std;

        /**
         * Constructs a ReferenceBlock3D object with the specified parameters.
         *
         * @param pixels A 1D array of float values containing the normalized block pixels.
         * @param width Block width in pixels.
         * @param height Block height in pixels.
         * @param radiusWidth Block radius along its width in pixels.
         * @param radiusHeight Block radius along its height in pixels.
         * @param size Total number of pixels in the block.
         * @param mean Mean value of the block pixels.
         * @param std Standard deviation of the block pixels.
         */
        public ReferenceBlock3D(float[] pixels, int width, int height, int depth, int radiusWidth,
                                int radiusHeight, int radiusDepth, int size, float mean, float std){

            this.pixels = pixels;
            this.width = width;
            this.height = height;
            this.depth = depth;
            this.radiusWidth = radiusWidth;
            this.radiusHeight = radiusHeight;
            this.radiusDepth = radiusDepth;
            this.size = size;
            this.mean = mean;
            this.std = std;
        }


        /**
         * Returns the normalized block pixels.
         *
         * @return A 1D array of float values containing the normalized block pixels retained
         *         after processing.
         */
        public float[] getPixels() {
            return pixels;
        }

        /**
         * Returns the block width in pixels.
         *
         * @return Block width.
         */
        public int getWidth() {
            return width;
        }

        /**
         * Returns the block height in pixels.
         *
         * @return Block height.
         */
        public int getHeight() {
            return height;
        }

        /**
         * Returns the block depth in pixels.
         *
         * @return Block height.
         */
        public int getDepth() {
            return depth;
        }

        /**
         * Returns the block radius along its width in pixels.
         *
         * @return Block width radius.
         */
        public int getRadiusWidth() {
            return radiusWidth;
        }

        /**
         * Returns the block radius along its height in pixels.
         *
         * @return Block height radius.
         */
        public int getRadiusHeight() {
            return radiusHeight;
        }

        /**
         * Returns the block radius along its height in pixels.
         *
         * @return Block height radius.
         */
        public int getRadiusDepth() {
            return radiusDepth;
        }

        /**
         * Returns the total number of pixels in the block.
         *
         * @return Size of the block.
         */
        public int getSize() {
            return size;
        }

        /**
         * Returns the mean value of the block pixels.
         *
         * @return Mean of the block.
         */
        public float getMean() {
            return mean;
        }

        /**
         * Returns the standard deviation of the block pixels.
         *
         * @return Standard deviation of the block.
         */
        public float getStd() {
            return std;
        }
    }

    /**
     * Custom object to hold a 3D input image along with its dimensions and statistics.
     */
    public static class InputImage3D
    {
        private final float[] imageArray;
        private final int width;
        private final int height;
        private final int depth;
        private final int size;
        private final Calibration calibration;
        private final float gain;
        private final float sigma;
        private final float offset;

        /**
         * Constructs an InputImage2D object with the specified image data and dimensions.
         *
         * @param imageArray A 1D array of float values representing the input image pixels.
         * @param width The width of the image in pixels.
         * @param height The height of the image in pixels.
         * @param size The total number of pixels in the image.
         */
        public InputImage3D(float[] imageArray, int width, int height, int depth, int size, Calibration calibration,
                            float gain, float sigma, float offset)
        {
            this.imageArray = imageArray;
            this.width = width;
            this.height = height;
            this.depth = depth;
            this.size = size;
            this.calibration = calibration;
            this.gain = gain;
            this.sigma = sigma;
            this.offset = offset;
        }

        /**
         * Returns the input image pixels as a 1D array.
         *
         * @return A 1D array of float values containing the input image pixels.
         */
        public float[] getImageArray() {
            return imageArray;
        }

        /**
         * Returns the width of the image in pixels.
         *
         * @return The width of the image.
         */
        public int getWidth() {
            return width;
        }

        /**
         * Returns the height of the image in pixels.
         *
         * @return The height of the image.
         */
        public int getHeight() {
            return height;
        }

        /**
         * Returns the depth of the image in pixels.
         *
         * @return The height of the image.
         */
        public int getDepth() {
            return depth;
        }

        /**
         * Returns the total number of pixels in the image.
         *
         * @return The size of the image.
         */
        public int getSize() {
            return size;
        }

        /**
         * Returns the ImageJ calibration object of the image.
         *
         * @return The calibration object of the image.
         */
        public Calibration getCalibration(){
            return calibration;
        }

        /**
         * Returns the optimised "gain" parameter used in the VST.
         *
         * @return The calibration object of the image.
         */
        public float getGain(){
            return gain;
        }

        /**
         * Returns the optimised "sigma" parameter used in the VST.
         *
         * @return The calibration object of the image.
         */
        public float getSigma(){
            return sigma;
        }

        /**
         * Returns the optimised "offset" parameter used in the VST.
         *
         * @return The calibration object of the image.
         */
        public float getOffset(){
            return offset;
        }
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 2D ---- //
    // ----------------------------------------- //

    /**
     * Retrieves a ReferenceBlock2D object from a 2D image containing a reference block.
     *
     * <p>If the blockID or the block's dimensions are invalid (e.g., not odd), the function will return {@code null}.</p>
     * <p>The block image must be 8-bit, 16-bit, or 32-bit; RGB is not supported.</p>
     *
     * @param blockID The ImageJ/Fiji window ID of the image containing the reference block.
     * @return A ReferenceBlock2D object, or {@code null} if the input image is invalid (i.e., window not found or even block dimensions).
     */
    public static ReferenceBlock2D getReferenceBlock2D(int blockID)
    {
        ImagePlus imp = WindowManager.getImage(blockID);

        if (imp == null) {
            IJ.error("Block image not found. Try again.");
            return null;
        }

        ImageProcessor ip = imp.getProcessor();
        FloatProcessor fp = ip.convertToFloatProcessor();
        float[] blockProcessor = (float[]) fp.getPixels();
        int bW = fp.getWidth(); // Block width
        int bH = fp.getHeight(); // Block height
        IJ.log("Block dimensions: " + bW + "x" + bH);

        // Check if block dimensions are odd; return null if not
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Block dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return null;
        }

        // Calculate block radius
        int bRW = bW / 2; // Patch radius (x-axis)
        int bRH = bH / 2; // Patch radius (y-axis)

        // Get final block size (after removing pixels outside inbound circle/ellipse)
        int blockSize = 0;
        for (int j = 0; j < bH; j++) {
            for (int i = 0; i < bW; i++) {
                float dx = (float) (i - bRW);
                float dy = (float) (j - bRH);
                if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        // Get flattened block array (keeping only the pixels within the inbound circle/ellipse)
        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int j = 0; j < bH; j++) {
            for (int i = 0; i < bW; i++) {
                float dx = (float) (i - bRW);
                float dy = (float) (j - bRH);
                if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) <= 1.0f) {
                    blockArray[index] = blockProcessor[j * bW + i];
                    index++;
                }
            }
        }

        // Calculate block min and max
        float blockMin = Float.MAX_VALUE; // Initialize as a very large number
        float blockMax = -Float.MAX_VALUE; // Initialize as a very small number

        for (int i = 0; i < blockSize; i++) {
            blockMin = Math.min(blockMin, blockArray[i]);
            blockMax = Math.max(blockMax, blockArray[i]);
        }

        // Normalize and calculate mean
        float blockMean = 0.0f;
        for (int i = 0; i < blockSize; i++) {
            blockArray[i] = (blockArray[i] - blockMin) / (blockMax - blockMin + EPSILON);
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        // Subtract mean
        for (int i = 0; i < blockSize; i++) {
            blockArray[i] = blockArray[i] - blockMean;
        }

        // Calculate block standard deviation
        float blockStd = 0.0f;
        for (int i = 0; i < blockSize; i++) {
            blockStd += (blockArray[i] - blockMean) * (blockArray[i] - blockMean);
        }
        blockStd = (float) Math.sqrt(blockStd / ((float) (blockSize - 1)));

        return new ReferenceBlock2D(blockArray, bW, bH, bRW, bRH, blockSize, blockMean, blockStd);
    }


    /**
     * Retrieves an InputImage2D object from a 2D image, optionally stabilizing noise variance
     * and normalizing the output.
     *
     * <p>If the imageID is invalid, the function will return {@code null}.</p>
     *
     * @param imageID The ImageJ/Fiji window ID of the image to be retrieved.
     * @param stabiliseNoiseVariance If {@code true}, applies variance stabilization to the image.
     * @param normalizeOutput If {@code true}, normalizes the pixel values of the image.
     * @return An InputImage2D object, or {@code null} if the input image is invalid (i.e., window not found).
     */
    public static InputImage2D getInputImage2D(int imageID, boolean stabiliseNoiseVariance, String gatMethod, int maxIter, boolean normalizeOutput)
    {
        // Get ImagePlus object
        ImagePlus imp = WindowManager.getImage(imageID);

        // Check if image dimensions are odd; return null if not found
        if (imp == null) {
            IJ.error("Image not found. Try again.");
            return null;
        }

        // Get processors and dimensions
        ImageProcessor ip = imp.getProcessor();
        FloatProcessor fp = ip.convertToFloatProcessor();
        float[] imageArray = (float[]) fp.getPixels();
        int width = fp.getWidth();
        int height = fp.getHeight();
        int size = width * height;
        IJ.log("Image dimensions: "+width+"x"+height);

        // Variance-stabilizing transform - Simplex method
        if (stabiliseNoiseVariance && gatMethod == "Simplex") {
            IJ.log("Stabilizing noise variance of the image...");
            GATMinimizer2D minimizer = new GATMinimizer2D(imageArray, width, height, 1, 10, 100, maxIter);
            minimizer.run();
            float gain = (float) minimizer.gain;
            float sigma = (float) minimizer.sigma;
            float offset = (float) minimizer.offset;

            imageArray = VarianceStabilisingTransform2D_.getGAT(imageArray, gain, sigma, offset);
        }

        // Variance-stabilizing transform - Quad/Octree method
        if (stabiliseNoiseVariance && gatMethod == "Quad/Octree") {
            IJ.log("Stabilizing noise variance of the image...");

            // Build Quadtree
            QuadTree_ quadTree = new QuadTree_(width, height, 4, 0.01f); // TODO: dont hardcode these params
            quadTree.buildTree(imageArray);

            // Calculate robust mean and variance estimations from the Quadtree nodes
            quadTree.calculateRobustMeans(imageArray, 50); // TODO: dont hardcode these params
            quadTree.calculateLTSVariances(imageArray, 0.75f); // TODO: dont hardcode these params
            List<double[]> meanVariancePairs = quadTree.collectMeanVariancePairs();

            // Perform linear regression to calculate g0 and eDC
            double[] regression = quadTree.performLinearRegression(meanVariancePairs);
            double g0 = regression[0];
            double eDC = regression[1];

            // Apply GAT
            imageArray = quadTree.applyGATtree(imageArray, width*height, g0, eDC);
        }

        // Normalize output (we always do this but I coded this in case we want to change)
        if (normalizeOutput) {
            // Get min and max
            float imageMin = Float.MAX_VALUE;
            float imageMax = -Float.MAX_VALUE;
            for (int i = 0; i < size; i++) {
                float pixelValue = imageArray[i];
                imageMin = Math.min(imageMin, pixelValue);
                imageMax = Math.max(imageMax, pixelValue);
            }

            // Normalize
            for (int i = 0; i < size; i++) {
                imageArray[i] = (imageArray[i] - imageMin) / (imageMax - imageMin + EPSILON);
            }
        }

        return new InputImage2D(imageArray, width, height, size);
    }

    /**
     * Retrieves an InputImage2D object from a 2D image, optionally stabilizing noise variance
     * and normalizing the output.
     *
     * NOTE: This is an OVERLOAD method that takes an image array instead of an ImageJ image ID and uses predefined
     * GAT parameter values to calculate the VST.
     *
     * <p>If the imageID is invalid, the function will return {@code null}.</p>
     *
     * @param imageArray An image array (float).
     * @param imageWidth The width of the image.
     * @param imageHeight The height of the image.
     * @param stabiliseNoiseVariance If {@code true}, applies variance stabilization to the image.
     * @param gain The gain to be used in the VST
     * @param sigma The sigma to be used in the VST
     * @param offset The offset to be used in the VST
     * @param normalizeOutput If {@code true}, normalizes the pixel values of the image.
     * @return An InputImage2D object, or {@code null} if the input image is invalid (i.e., window not found).
     */
    public static InputImage2D getInputImage2D(float[] imageArray, int imageWidth, int imageHeight,
                                               boolean stabiliseNoiseVariance, float gain, float sigma, float offset,
                                               boolean normalizeOutput)
    {

        int imageSize = imageWidth * imageHeight;

        if (stabiliseNoiseVariance) {
            IJ.log("Stabilizing noise variance of the image...");
            imageArray = VarianceStabilisingTransform2D_.getGAT(imageArray, gain, sigma, offset);
        }

        if (normalizeOutput) {
            // Get min and max
            float imageMin = Float.MAX_VALUE;
            float imageMax = -Float.MAX_VALUE;
            for (int i = 0; i < imageSize; i++) {
                float pixelValue = imageArray[i];
                imageMin = Math.min(imageMin, pixelValue);
                imageMax = Math.max(imageMax, pixelValue);
            }

            // Normalize
            for (int i = 0; i < imageSize; i++) {
                imageArray[i] = (imageArray[i] - imageMin) / (imageMax - imageMin + EPSILON);
            }
        }

        return new InputImage2D(imageArray, imageWidth, imageHeight, imageSize);
    }


    /**
     * Calculates the mean noise variance for a 2D reference pixel array.
     *
     * <p>This method divides the reference pixel array into blocks and computes local variances.
     * It then returns the average of the lowest 3% of these variances, scaled based on a specified formula.</p>
     *
     * @param refPixels A 1D array (float32) containing the reference pixel values.
     * @param w The width of the image (in pixels).
     * @param h The height of the image (in pixels).
     * @param wh The total number of pixels in the image (width * height).
     * @return The calculated mean noise variance.
     */
    public static float getMeanNoiseVar2D(float[] refPixels, int w, int h, int wh)
    {
        int blockWidth, blockHeight;
        int CIF = 352 * 288; // Resolution of a CIF file

        // Determine block dimensions based on image size
        if (wh <= CIF) {
            blockWidth = 8;
            blockHeight = 8;
        } else {
            blockWidth = 16;
            blockHeight = 16;
        }

        // Calculate the number of blocks in each dimension
        int nBlocksX = w / blockWidth; // number of blocks in each row
        int nBlocksY = h / blockHeight; // number of blocks in each column
        int nBlocks = nBlocksX * nBlocksY; // total number of blocks
        float[] localVars = new float[nBlocks];
        Arrays.fill(localVars, 0.0f);

        // Calculate local variances
        int index = 0;
        for (int y = 0; y < nBlocksY; y++) {
            for (int x = 0; x < nBlocksX; x++) {
                float[] meanVar = getMeanAndVarBlock2D(refPixels, w, x*blockWidth, y*blockHeight, (x+1)*blockWidth, (y+1)*blockHeight);
                localVars[index] = meanVar[1]; // Store variance
                index++;
            }
        }

        // Sort the local variances
        Arrays.sort(localVars);

        // Get the 3% lowest variances and calculate their average
        int nVars = (int) (0.03f * (float) nBlocks + 1.0f); // Number of blocks corresponding to 3% of the total
        float noiseVar = 0.0f;

        for (int i = 0; i < nVars; i++) {
            noiseVar += localVars[i];
        }

        // Calculate average of the lowest variances
        noiseVar = Math.abs(noiseVar / (float) nVars);
        noiseVar = (1.0f + 0.001f * (noiseVar - 40.0f)) * noiseVar;

        return noiseVar;
    }


    /**
     * Calculates the mean and variance of a block of pixels from a 1D pixel array in a single pass.
     *
     * <p>This method computes the mean and variance for a specified rectangular block of pixels
     * defined by its start and end coordinates.</p>
     *
     * @param imageArray A 1D array (float32) containing the pixel values.
     * @param imageWidth The width of the image (in pixels), used for indexing into the pixel array.
     * @param xStart The starting x-coordinate (inclusive) of the block.
     * @param yStart The starting y-coordinate (inclusive) of the block.
     * @param xEnd The ending x-coordinate (exclusive) of the block.
     * @param yEnd The ending y-coordinate (exclusive) of the block.
     * @return A float array where the first element is the mean and the second element is the variance of the block.
     */
    public static float[] getMeanAndVarBlock2D(float[] imageArray, int imageWidth, int xStart, int yStart, int xEnd,
                                               int yEnd) {
        // Validate block dimensions
        int blockWidth = xEnd - xStart;
        int blockHeight = yEnd - yStart;
        int blockSize = blockWidth * blockHeight;

        if (blockWidth <= 0 || blockHeight <= 0) {
            throw new IllegalArgumentException("Block dimensions must be positive.");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("Block size cannot be zero.");
        }

        // Validate input array dimensions
        if (xStart < 0 || yStart < 0 || xEnd > imageWidth || yEnd > imageArray.length / imageWidth) {
            throw new IllegalArgumentException("Block coordinates are out of bounds.");
        }

        // Initialize variables
        float mean = 0.0f;
        float sq_sum = 0.0f;

        // Compute mean and sum of squares
        for (int y = yStart; y < yEnd; y++) {
            for (int x = xStart; x < xEnd; x++) {
                float v = imageArray[y * imageWidth + x];
                // Validate input pixel value
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    throw new IllegalArgumentException("Input contains invalid (NaN or Infinite) values.");
                }
                mean += v;
                sq_sum += v * v;
            }
        }

        // Compute final mean and variance
        mean /= blockSize; // Prevent division by zero
        float var = sq_sum / blockSize - mean * mean; // Variance formula

        // Ensure result is valid
        if (Float.isNaN(var)) {
            throw new IllegalStateException("Computed variance is NaN. Check inputs and computations.");
        }

        return new float[]{mean, var};
    }


    /**
     * Apply a mask to an image (in-place).
     *
     * <p>This method modifies the input image array by applying a provided mask,
     * which scales each pixel in the image according to the corresponding value in the mask.</p>
     *
     * @param imageArray An input image array (float32).
     * @param imageWidth The width of the input image array.
     * @param imageHeight The height of the input image array.
     * @param mask A user-provided mask (float32).
     * @return A masked image array (the modified input image array).
     */
    public static float[] applyMask2D(float[] imageArray, int imageWidth, int imageHeight, float[] mask)
    {
        // Apply mask
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                imageArray[y * imageWidth + x] *= mask[y * imageWidth + x];
            }
        }

        return imageArray;
    }


    /**
     * Calculate relevance mask based on structural relevance.
     *
     * <p>This method generates a binary mask where the center pixels of each local neighborhood are assigned a value of zero
     * if the local variance is below or equal to the relevance threshold, which is determined by the mean noise variance
     * multiplied by a user-defined relevance constant.</p>
     *
     * @param imageArray An input image array (float) used to calculate the mean noise variance and the relevance mask.
     * @param imageWidth The width of the input image array.
     * @param imageHeight The height of the input image array.
     * @param blockRadiusWidth The width radius of the block used in the repetition analysis.
     * @param blockRadiusHeight The height radius of the block used in the repetition analysis.
     * @param localStds The local standard deviations array calculated with {@link CLUtils.getLocalStatistics2D()}.
     * @param relevanceConstant A user-provided relevance constant to control the strength of the relevance filter.
     * @return A {@link RelevanceMask} object.
     */
    public static RelevanceMask getRelevanceMask(float[] imageArray, int imageWidth, int imageHeight,
                                                 int blockRadiusWidth, int blockRadiusHeight, float[] localStds,
                                                 float relevanceConstant)
    {
        // Calculate noise mean variance
        int imageSize = imageWidth * imageHeight;

        float noiseMeanVariance = getMeanNoiseVar2D(imageArray, imageWidth, imageHeight, imageSize);
        float relevanceThreshold = noiseMeanVariance*relevanceConstant;

        // Calculate relevance mask
        float[] relevanceMask = new float[imageSize];
        Arrays.fill(relevanceMask, 0.0f); // Fill with zeros just to be sure
        for (int y = blockRadiusHeight; y < imageHeight - blockRadiusHeight; y++) {
            for (int x = blockRadiusWidth; x < imageWidth - blockRadiusWidth; x++) {
                if ((localStds[y * imageWidth + x] * localStds[y * imageWidth + x]) <= relevanceThreshold) {
                    relevanceMask[y * imageWidth + x] = 0.0f; // Assign zero if local variance is below threshold
                } else {
                    relevanceMask[y * imageWidth + x] = 1.0f; // Assign one if local variance is above threshold
                }
            }
        }

        return new RelevanceMask(relevanceMask, relevanceConstant, relevanceThreshold, noiseMeanVariance); // Return the relevance mask
    }


    /**
     * Normalize an image (in-place) to a specified range, with the option to avoid masked pixels.
     *
     * <p>This method calculates the minimum and maximum pixel values of the input image while excluding
     * any masked pixels, and then normalizes the image based on these values.</p>
     *
     * @param imageArray The input image array (float) to normalize.
     * @param imageWidth The width of the input image array.
     * @param imageHeight The height of the input image array.
     * @param borderWidth The width of the border to exclude from the normalization calculation.
     * @param borderHeight The height of the border to exclude from the normalization calculation.
     * @param mask A binary mask (float) where masked pixels will be excluded from the calculation.
     * @return A normalized image array (float) where pixel values are remapped to the range [0, 1].
     */
    public static float[] normalizeImage2D(float[] imageArray, int imageWidth, int imageHeight, int borderWidth,
                                           int borderHeight, float[] mask)
    {
        // Find min and max
        float imageMin = Float.MAX_VALUE;
        float imageMax = -Float.MAX_VALUE;

        if(mask==null) {
            for (int y=borderHeight; y<imageHeight-borderHeight; y++) {
                for (int x=borderWidth; x<imageWidth-borderWidth; x++) {
                    imageMin = min(imageArray[y*imageWidth+x], imageMin);
                    imageMax = max(imageArray[y*imageWidth+x], imageMax);
                }
            }
        }else{
            for (int y=borderHeight; y<imageHeight-borderHeight; y++) {
                for (int x=borderWidth; x<imageWidth-borderWidth; x++) {
                    if (mask[y*imageWidth+x] != 0.0f && mask[y*imageWidth+x] != 1.0f) {
                        throw new IllegalArgumentException("Relevance mask must be binary (0.0f or 1.0f)");
                    }
                    if (mask[y*imageWidth+x] > 0.0f) {
                        imageMin = min(imageArray[y*imageWidth+x], imageMin);
                        imageMax = max(imageArray[y*imageWidth+x], imageMax);
                    }
                }
            }
        }

        // Remap pixels
        float[] normalizedArray = imageArray.clone();
        if(mask==null) {
            for (int y=borderHeight; y<imageHeight-borderHeight; y++) {
                for (int x=borderWidth; x<imageWidth-borderWidth; x++) {
                        normalizedArray[y*imageWidth+x] = (normalizedArray[y*imageWidth+x]-imageMin)/(imageMax-imageMin+EPSILON);
                }
            }
        }else{
            for (int y=borderHeight; y<imageHeight-borderHeight; y++) {
                for (int x=borderWidth; x<imageWidth-borderWidth; x++) {
                    if (mask[y*imageWidth+x]>0.0f) {
                        normalizedArray[y*imageWidth+x] = (normalizedArray[y*imageWidth+x]-imageMin)/(imageMax-imageMin+EPSILON);
                    }
                }
            }
        }
        return normalizedArray;
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 3D ---- //
    // ----------------------------------------- //

    /**
     * Retrieves a ReferenceBlock3D object from a 3D image containing a reference block.
     *
     * <p>If the blockID or the block's dimensions are invalid (e.g., not odd), the function will return {@code null}.</p>
     * <p>The block image must be 8-bit, 16-bit, or 32-bit; RGB is not supported.</p>
     *
     * @param blockID The ImageJ/Fiji window ID of the image containing the reference block.
     * @return A ReferenceBlock2D object, or {@code null} if the input image is invalid (i.e., window not found or even block dimensions).
     */
    public static ReferenceBlock3D getReferenceBlock3D(int blockID)
    {
        ImagePlus imp = WindowManager.getImage(blockID);

        // Check if image is found
        if (imp == null) {
            IJ.error("Block image not found. Try again.");
            throw new IllegalArgumentException("Block image not found.");
        }

        // Get block dimensions
        ImageStack ims = imp.getStack();
        int bW = ims.getWidth(); // Block width
        int bH = ims.getHeight(); // Block height
        int bZ = ims.getSize(); // block depth
        int bRW = bW / 2; // Patch radius (x-axis)
        int bRH = bH / 2; // Patch radius (y-axis)
        int bRZ = bZ / 2; // Patch radius (z-axis)
        IJ.log("Block dimensions: " + bW + "x" + bH + "x" + bZ);

        // Check if block dimensions are odd
        if (bW % 2 == 0 || bH % 2 == 0 || bZ % 2 == 0) {
            IJ.error("Block dimensions must be odd (e.g., 3x3x3 or 5x5x5). Please try again.");
            throw new IllegalArgumentException("Block dimensions must be odd.");
         }

        // Check if patch has at least 3 slices
        if (bZ < 3) {
            IJ.error("Block must have at least 3 slices. Please try again.");
            throw new IllegalArgumentException("Reference block must have at least 3 slices.");
        }

        // Get final block size (after removing pixels outside the sphere/ellipsoid)
        int blockSize = 0;
        for (int z = 0; z < bZ; z++) {
            for (int y = 0; y < bH; y++) {
                for (int x = 0; x < bW; x++) {
                    float dx = (float) (x - bRW);
                    float dy = (float) (y - bRH);
                    float dz = (float) (z - bRZ);
                    if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) + ((dz * dz) / (float) (bRZ * bRZ)) <= 1.0f) {
                        blockSize++;
                    }
                }
            }
        }

        // Get block array
        float[][] blockArray = new float[bZ][bW * bH];
        for (int z = 0; z < bZ; z++) {
            for (int y = 0; y < bH; y++) {
                for (int x = 0; x < bW; x++) {
                    blockArray[z][y*bW+x] = ims.getProcessor(z+1).convertToFloatProcessor().getf(x,y);
                }
            }
        }

        // Get flattened block array (keeping only the pixels within the inbound spheroid)
        float[] blockArray1D = new float[blockSize];
        int index = 0;
        for (int z=0; z<bZ; z++) {
            for (int y=0; y<bH; y++) {
                for (int x=0; x<bW; x++) {
                    float dx = (float)(x-bRW);
                    float dy = (float)(y-bRH);
                    float dz = (float)(z-bRZ);
                    if (((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f) {
                        blockArray1D[index] = blockArray[z][y*bW+x];
                        index++;
                    }
                }
            }
        }

        // Normalize
        blockArray1D = normalizeArray(blockArray1D);

        // Calculate mean
        float blockMean = 0.0f;
        for (int i=0; i<blockSize; i++) {
            blockMean += blockArray1D[i];
        }
        blockMean /= (float) blockSize;

        // Subtract mean
        for (int i=0; i<blockSize; i++) {
            blockArray1D[i] = blockArray1D[i] - blockMean;
        }

        // Normalize again
        blockArray1D = normalizeArray(blockArray1D);

        // Calculate block mean and standard deviation
        blockMean = 0.0f;
        for (int i=0; i<blockSize; i++) {
            blockMean += blockArray1D[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for (int i = 0; i < blockSize; i++) {
            blockStd += (blockArray1D[i] - blockMean) * (blockArray1D[i] - blockMean);
        }
        blockStd = (float) Math.sqrt(blockStd / ((float) (blockSize - 1)));

        return new ReferenceBlock3D(blockArray1D, bW, bH, bZ, bRW, bRH, bRZ, blockSize, blockMean, blockStd);
    }


    /**
     * Retrieves an InputImage3D object from a 3D image, optionally stabilizing noise variance
     * and normalizing the output.
     *
     * <p>If the imageID is invalid, the function will return {@code null}.</p>
     *
     * @param imageID The ImageJ/Fiji window ID of the image to be retrieved.
     * @param stabiliseNoiseVariance If {@code true}, applies variance stabilization to the image.
     * @param normalizeOutput If {@code true}, normalizes the pixel values of the image.
     * @return An InputImage3D object, or {@code null} if the input image is invalid (i.e., window not found).
     */
    public static InputImage3D getInputImage3D(int imageID, boolean stabiliseNoiseVariance, String gatMethod, boolean normalizeOutput)
    {
        // Get ImagePlus object
        ImagePlus imp = WindowManager.getImage(imageID);

        // Check if image dimensions are odd; return null if not found
        if (imp == null) {
            IJ.error("Image not found. Try again.");
            return null;
        }

        // Get calibration parameters
        Calibration calibration = imp.getCalibration();

        // Get ImageStack object
        ImageStack ims = imp.getStack();

        // Get processors and dimensions
        //float[] imageArray = (float[]) fp.getPixels();
        int width = ims.getWidth();
        int height = ims.getHeight();
        int depth = ims.getSize();
        int size = width * height * depth;
        IJ.log("Image dimensions: "+width+"x"+height+"x"+depth);

        // Check if image has at least 3 slices, otherwise kill program
        if (depth < 3) {
            IJ.error("Image must have at least 3 slices. Please try again.");
            return null;
        }

        // Get image stack array (float[z][wh]) // TODO: ONLY REQUIRED FOR GATMINIMIZER
        float[][] stackArray = new float[depth][width*height];
        float[] imageArray = new float[width*height*depth];
        for(int z=0; z<depth; z++) {
            FloatProcessor fp = ims.getProcessor(z+1).convertToFloatProcessor();
            for (int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    stackArray[z][y*width+x] = fp.getf(x,y); // getPixels() is easier but isn't working well
                    imageArray[width*height*z+y*width+x] = fp.getf(x, y);
                }
            }
        }

        // Stabilise noise variance
        float gain = 0.0f; // Placeholder
        float sigma = 0.0f; // Placeholder
        float offset = 0.0f; // Placeholder

        if (stabiliseNoiseVariance && gatMethod == "Simplex") {

            IJ.log("Stabilizing noise variance of the image...");

            // Optimise GAT parameters
            int maxIter = 5000; // TODO: DO NOT HARDCODE THIS
            GATMinimizer3D minimizer = new GATMinimizer3D(imageArray, width, height, depth, 0, 100, 0, maxIter);
            minimizer.run();

            gain = (float) minimizer.gain;
            sigma = (float) minimizer.sigma;
            offset = (float) minimizer.offset;

            // Get variance-stabilised image with optimised parameters
            for (int z=0; z<depth; z++) {
                stackArray[z] = VarianceStabilisingTransform2D_.getGAT(imageArray, minimizer.gain, minimizer.sigma, minimizer.offset);
            }
        }

        if (stabiliseNoiseVariance && gatMethod == "Quad/Octree") {

            IJ.log("Stabilizing noise variance of the image...");

            // Build Quadtree
            OcTree_ ocTree = new OcTree_(width, height, depth, 4, 0.01f); // TODO: dont hardcode these params
            ocTree.buildTree(imageArray);

            // Calculate robust mean and variance estimations from the Quadtree nodes
            ocTree.calculateRobustMeans(imageArray, 50); // TODO: dont hardcode these params
            ocTree.calculateLTSVariances(imageArray, 0.75f); // TODO: dont hardcode these params
            List<double[]> meanVariancePairs = ocTree.collectMeanVariancePairs();

            // Perform linear regression to calculate g0 and eDC
            float[] regression = ocTree.performLinearRegression(meanVariancePairs);
            float g0 = regression[0];
            float eDC = regression[1];

            // Apply GAT
            imageArray = ocTree.applyGATtree(imageArray, width*height, g0, eDC);
        }

        // Normalise to range
        if (normalizeOutput) {
            imageArray = normalizeArray(imageArray);
        }

        return new InputImage3D(imageArray, width, height, depth, size, calibration, gain, sigma, offset);
    }


    /**
     * Retrieves an InputImage3D object from a 3D image, optionally stabilizing noise variance
     * and normalizing the output.
     *
     * NOTE: This is an OVERLOAD method that takes an image array instead of an ImageJ image ID.
     *
     * <p>If the imageID is invalid, the function will return {@code null}.</p>
     *
     * @param imageArray An image array of the image (float).
     * @param stabiliseNoiseVariance If {@code true}, applies variance stabilization to the image.
     * @param normalizeOutput If {@code true}, normalizes the pixel values of the image.
     * @return An InputImage3D object, or {@code null} if the input image is invalid (i.e., window not found).
     */
    public static InputImage3D getInputImage3D(float[] imageArray, int width, int height, int depth, boolean stabiliseNoiseVariance, boolean normalizeOutput)
    {

        int size = width*height*depth;

        // Stabilise noise variance
        float gain = 0.0f; // Placeholder
        float sigma = 0.0f; // Placeholder
        float offset = 0.0f; // Placeholder

        if (stabiliseNoiseVariance) {
            IJ.log("Stabilizing noise variance of the image...");

            // Get image stack array (float[z][wh])
            float[][] stackArray = new float[depth][width*height];
            for(int z=0; z<depth; z++) {
                for (int y=0; y<height; y++) {
                    for (int x=0; x<width; x++) {
                        stackArray[z][y*width+x] = imageArray[width*height*z+y*width+x];
                    }
                }
            }

            // Optimise GAT parameters
            int maxIter = 5000; // TODO: DO NOT HARDCODE THIS
            GATMinimizer3D minimizer = new GATMinimizer3D(imageArray, width, height, depth, 0, 100, 0, maxIter);
            minimizer.run();
            gain = (float) minimizer.gain;
            sigma = (float) minimizer.sigma;
            offset = (float) minimizer.offset;

            // Get variance-stabilised image with optimised parameters
            for (int z=0; z<depth; z++) {
                stackArray[z] = VarianceStabilisingTransform2D_.getGAT(stackArray[z], minimizer.gain, minimizer.sigma, minimizer.offset);
            }

            // Get stack array (1D)
            float[] stackArray1D = new float[size];
            for(int z=0; z<depth; z++) {
                for (int y=0; y<height; y++) {
                    for (int x=0; x<width; x++) {
                        imageArray[width*height*z+y*width+x] = stackArray[z][y*width+x];
                    }
                }
            }

            // Normalise to range
            if (normalizeOutput) {
                imageArray = normalizeArray(stackArray1D);
            }
        } else if (normalizeOutput) {
            imageArray = normalizeArray(imageArray);
        }

        Calibration calibration = null;

        return new InputImage3D(imageArray, width, height, depth, size, calibration, gain, sigma, offset);
    }


    /**
     * Calculates the mean and variance of a block of pixels from a 1D pixel array of a 3D image with a single pass.
     *
     * <p>This method computes the mean and variance for a specified rectangular block of pixels
     * defined by its start and end coordinates.</p>
     *
     * @param pixels A 1D array (float32) containing the pixel values.
     * @param imageWidth The width of the image (in pixels), used for indexing into the pixel array.
     * @param imageHeight The height of the image (in pixels), used for indexing into the pixel array.
     * @param xStart The starting x-coordinate (inclusive) of the block.
     * @param yStart The starting y-coordinate (inclusive) of the block.
     * @param zStart The starting z-coordinate (incusive) of the block.
     * @param xEnd The ending x-coordinate (exclusive) of the block.
     * @param yEnd The ending y-coordinate (exclusive) of the block.
     * @param zEnd The ending z-coordinate (exclusive) of the block.
     * @return A float array where the first element is the mean and the second element is the variance of the block.
     */
    public static double[] getMeanAndVarBlock3D(float[] pixels, int imageWidth, int imageHeight, int xStart, int yStart,
                                               int zStart, int xEnd, int yEnd, int zEnd)
    {
        double mean = 0.0f;
        double var;
        double sq_sum = 0.0f;

        int blockWidth = xEnd - xStart; // Block width
        int blockHeight = yEnd - yStart; // Block height
        int blockDepth = zEnd - zStart; // Block depth
        int blockSize = blockWidth * blockHeight * blockDepth; // Total number of pixels in the block

        // Calculate the sum of pixel values and the sum of squared pixel values
        for(int z=zStart; z<zEnd; z++)
            for (int y=yStart; y<yEnd; y++) {
                for (int x=xStart; x<xEnd; x++) {
                    double v = (double)pixels[imageWidth*imageHeight*z+y*imageWidth+x]; // Get the pixel value
                    mean += v; // Accumulate the sum
                    sq_sum += v * v; // Accumulate the sum of squares
                }
            }

        mean = mean / blockSize; // Calculate the mean
        var = sq_sum / blockSize - mean * mean; // Calculate the variance

        return new double[]{mean, var}; // Return mean and variance
    }


    /*
    /**
     * Calculates the mean noise variance for a 3D reference pixel array.
     *
     * <p>This method divides the reference pixel array into blocks and computes local variances.
     * It then returns the average of the lowest 3% of these variances, scaled based on a specified formula.</p>
     *
     * @param refPixels A flattened 1-D array (float32) containing the reference pixel values.
     * @param imageWidth The width of the image (in pixels).
     * @param imageHeight The height of the image (in pixels).
     * @param imageDepth the depth of the image (in pixels).
     * @param imageSize The total number of pixels in the image (width * height * depth).
     * @return The calculated mean noise variance.
     *
    public static float getMeanNoiseVar3D(float[] imageArray, int imageWidth, int imageHeight, int imageDepth, int imageSize) {
        int blockWidth, blockHeight, blockDepth;
        int CIF = 352 * 288; // Resolution of a CIF file

        // Determine block dimensions based on image size
        if (imageWidth*imageHeight <= CIF) {
            blockWidth = 8;
            blockHeight = 8;
            blockDepth = 5; // TODO: this was arbitrary, didn't want to use more slices
        } else {
            blockWidth = 16;
            blockHeight = 16;
            blockDepth = 5; // TODO: this was arbitrary, didn't want to use more slices
        }

        // Calculate the number of blocks in each dimension
        int nBlocksX = imageWidth / blockWidth; // number of blocks in each row
        int nBlocksY = imageHeight / blockHeight; // number of blocks in each column
        int nBlocksZ = imageDepth / blockDepth; // number of bocks in depth
        int nBlocks = nBlocksX * nBlocksY * nBlocksZ; // total number of blocks
        float[] localVars = new float[nBlocks];
        Arrays.fill(localVars, 0.0f);

        // Calculate local variances
        int index = 0;
        for(int z=0; z<nBlocksZ; z++) {
            for (int y=0; y<nBlocksY; y++) {
                for (int x=0; x<nBlocksX; x++) {
                    float[] meanVar = getMeanAndVarBlock3D(imageArray, imageWidth, imageHeight, imageDepth, x*blockWidth, y*blockHeight, z*blockDepth, (x+1)*blockWidth, (y+1)*blockHeight, (z+1)*blockDepth);
                    localVars[index] = meanVar[1]; // Store variance
                    index++;
                }
            }
        }
        // Sort the local variances
        Arrays.sort(localVars);

        // Get the 3% lowest variances and calculate their average
        int nVars = (int) (0.03f * (float) nBlocks + 1.0f); // Number of blocks corresponding to 3% of the total
        float noiseVar = 0.0f;

        for (int i = 0; i < nVars; i++) {
            noiseVar += localVars[i];
        }

        // Calculate average of the lowest variances
        noiseVar = Math.abs(noiseVar / (float) nVars);
        noiseVar = (1.0f + 0.001f * (noiseVar - 40.0f)) * noiseVar;

        return noiseVar;
    }
    */

    /**
     * TESTED WAY OF GETTING MEANNOISEVAR3D. SEE PREVIOUS METHOD FOR LIU ET AL IMPLEMENTATION - NEEDS TESTING
     * @param imageWidth
     * @param imageHeight
     * @param imageDepth
     * @param blockRadiusWidth
     * @param blockRadiusHeight
     * @param blockRadiusDepth
     * @param localStds
     * @return
     */
    public static float getMeanNoiseVar3D(int imageWidth, int imageHeight, int imageDepth, int blockRadiusWidth,
                                          int blockRadiusHeight, int blockRadiusDepth, float[] localStds)
    {
        float noiseMeanVar = 0.0f;
        float numElements = 0.0f;

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for(int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++){
                for(int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++){
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    noiseMeanVar += localStds[index] * localStds[index]; // Get local variance
                    numElements += 1.0f;
                }
            }
        }
        noiseMeanVar /= numElements;

        return noiseMeanVar;
    }

    /**
     * Calculate a 3D relevance mask based on structural relevance.
     *
     * <p>This method generates a 3D binary mask where the center pixels of each local neighborhood are assigned a value of zero
     * if the local variance is below or equal to the relevance threshold, which is determined by the mean noise variance
     * multiplied by a user-defined relevance constant.</p>
     *
     * @param imageWidth The width of the input image.
     * @param imageHeight The height of the input image.
     * @param imageDepth The depth of the input image.
     * @param blockRadiusWidth The width radius of the block used in the repetition analysis.
     * @param blockRadiusHeight The height radius of the block used in the repetition analysis.
     * @param blockRadiusDepth The depth radius of the block used in the repetition analysis.
     * @param localStds The local standard deviations array calculated with {@link CLUtils.getLocalStatistics3D()}.
     * @param relevanceConstant A user-provided relevance constant to control the strength of the relevance filter.
     * @return A binary mask (float) where center pixels of each local neighbourhood are assigned a value of zero if the local variance is below or equal
     *         to the relevance threshold; otherwise, they are assigned a value of one.
     */
    public static float[] getRelevanceMask3D(int imageWidth, int imageHeight, int imageDepth, int blockRadiusWidth,
                                             int blockRadiusHeight, int blockRadiusDepth, float[] localStds,
                                             float relevanceConstant)
    {
        // Calculate noise mean variance
        float noiseMeanVariance = getMeanNoiseVar3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                blockRadiusHeight, blockRadiusDepth, localStds);

        // Calculate relevance mask
        float[] relevanceMask = new float[imageWidth*imageHeight*imageDepth];
        Arrays.fill(relevanceMask, 0.0f); // Fill with zeros just to be sure

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth * imageHeight * z + y * imageWidth + x;
                    if ((localStds[index]*localStds[index]) <= noiseMeanVariance*relevanceConstant) {
                        relevanceMask[index] = 0.0f; // Assign zero if local variance is below threshold
                    } else {
                        relevanceMask[index] = 1.0f; // Assign one if local variance is above threshold
                    }
                }
            }
        }
        return relevanceMask; // Return the relevance mask
    }


    /**
     * Apply a mask to a 3-D image (in-place).
     *
     * <p>This method modifies the input image array by applying a provided mask,
     * which scales each pixel in the image according to the corresponding value in the mask.</p>
     *
     * @param imageArray An input image array (float32).
     * @param imageWidth The width of the input image.
     * @param imageHeight The height of the input image.
     * @param imageDepth The depth of the input image.
     * @param mask A user-provided binary mask (float32).
     * @return A masked image array (the modified input image array).
     */
    public static float[] applyMask3D(float[] imageArray, int imageWidth, int imageHeight, int imageDepth, float[] mask)
    {
        // Apply mask
        for(int z=0; z<imageDepth; z++) {
            for (int y=0; y<imageHeight; y++) {
                for (int x=0; x<imageWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    imageArray[index] *= mask[index];
                }
            }
        }

        return imageArray;
    }


    /**
     * Normalize a 3D image (in-place) to its range, with the option to avoid masked pixels.
     *
     * <p>This method calculates the minimum and maximum pixel values of the input image while excluding
     * any masked pixels, and then normalizes the image based on these values.</p>
     *
     * @param imageArray The input image array (float) to normalize.
     * @param imageWidth The width of the input image array.
     * @param imageHeight The height of the input image array.
     * @param borderWidth The width of the border to exclude from the normalization calculation.
     * @param borderHeight The height of the border to exclude from the normalization calculation.
     * @param mask A binary mask (float) where masked pixels will be excluded from the calculation.
     * @return A normalized image array (float) where pixel values are remapped to the range [0, 1].
     */
    public static float[] normalizeImage3D(float[] imageArray, int imageWidth, int imageHeight, int imageDepth, int borderWidth,
                                           int borderHeight, int borderDepth, float[] mask)
    {

        // Find min and max
        float imageMin = Float.MAX_VALUE;
        float imageMax = -Float.MAX_VALUE;

        if(mask==null) {
            for (int z = borderDepth; z < imageDepth - borderDepth; z++) {
                for (int y = borderHeight; y < imageHeight - borderHeight; y++) {
                    for (int x = borderWidth; x < imageWidth - borderWidth; x++) {
                        int index = imageWidth * imageHeight * z + y * imageWidth + x;
                        imageMin = min(imageArray[index], imageMin);
                        imageMax = max(imageArray[index], imageMax);
                    }
                }
            }
        }else{
            for (int z = borderDepth; z < imageDepth - borderDepth; z++) {
                for (int y = borderHeight; y < imageHeight - borderHeight; y++) {
                    for (int x = borderWidth; x < imageWidth - borderWidth; x++) {
                        int index = imageWidth * imageHeight * z + y * imageWidth + x;
                        if (mask[index] > 0.0f) {
                            imageMin = min(imageArray[index], imageMin);
                            imageMax = max(imageArray[index], imageMax);
                        }
                    }
                }
            }
        }

        // Remap pixels
        float[] normalizedImage = imageArray.clone();
        if(mask==null) {
            for (int z = borderDepth; z < imageDepth - borderDepth; z++) {
                for (int y = borderHeight; y < imageHeight - borderHeight; y++) {
                    for (int x = borderWidth; x < imageWidth - borderWidth; x++) {
                        int index = imageWidth * imageHeight * z + y * imageWidth + x;
                        normalizedImage[index] = (imageArray[index] - imageMin) / (imageMax - imageMin + EPSILON);
                    }
                }
            }
        }else{
            for (int z = borderDepth; z < imageDepth - borderDepth; z++) {
                for (int y = borderHeight; y < imageHeight - borderHeight; y++) {
                    for (int x = borderWidth; x < imageWidth - borderWidth; x++) {
                        int index = imageWidth * imageHeight * z + y * imageWidth + x;
                        if (mask[index] > 0.0f) {
                            normalizedImage[index] = (imageArray[index] - imageMin) / (imageMax - imageMin + EPSILON);
                        }
                    }
                }
            }
        }

        return normalizedImage;
    }


    // -------------------------- //
    // ---- UNSORTED METHODS ---- //
    // -------------------------- //

    /**
     * Normalizes an array to its range.
     *
     * @param array
     * @return A copy of the original array, normalized to the range
     */
    public static float[] normalizeArray(float[] array)
    {

        // Cache variables
        int arrayLength = array.length;

        // Get min max
        float arrayMin = Float.MAX_VALUE;
        float arrayMax = -Float.MAX_VALUE;
        for(int i=0; i<arrayLength; i++){
            arrayMin = min(array[i], arrayMin);
            arrayMax = max(array[i], arrayMax);
        }

        // Remap values
        float[] normalizedArray = new float[arrayLength];
        for(int i=0; i< array.length; i++){
            normalizedArray[i] = (array[i]-arrayMin) / (arrayMax-arrayMin+EPSILON);
        }
        return normalizedArray;
    }


    public static double getInvariant(float[] patch, int w, int h, int p, int q){
        // Get centroid x and y
        double moment_10 = 0.0f;
        double moment_01 = 0.0f;
        double moment_00 = 0.0f;
        for(int j=0; j<h; j++){
            for(int i=0; i<w; i++){
                moment_10 += patch[j*w+i] * i;
                moment_01 += patch[j*w+i] * j;
                moment_00 += patch[j*w+i];
            }
        }

        double centroid_x = moment_10 / (moment_00 + 0.000001f);
        double centroid_y = moment_01 / (moment_00 + 0.000001f);

        // Get mu_pq
        double mu_pq = 0.0f;
        for(int j=0; j<h; j++){
            for(int i=0; i<w; i++){
                mu_pq += patch[j*w+i] + pow(i+1-centroid_x, p) * pow(j+1-centroid_y, q);
            }
        }

        float invariant = (float) (mu_pq / pow(moment_00, (1+(p+q/2))));
        return invariant;
    }

    /**
     * Calculate the mean, variance, and standard deviation of an array in a single pass.
     *
     * <p>This method computes the mean, variance, and standard deviation of the input array.</p>
     *
     * @param a The input array of float values.
     * @return An array of float containing the mean, variance, and standard deviation in that order.
     *         Returns an array of zeros if the input array is empty.
     */
    private float[] meanVarStd(float[] a) {
        int n = a.length;
        if (n == 0) return new float[]{0, 0, 0}; // Return zeros for empty array

        double sum = 0.0;
        double sq_sum = 0.0;

        // Calculate sum and squared sum
        for (int i = 0; i < n; i++) {
            sum += a[i];
            sq_sum += a[i] * a[i];
        }

        double mean = sum / n; // Calculate mean
        double variance = Math.abs(sq_sum / n - mean * mean); // Calculate variance
        // abs() solves a bug where negative zeros appeared

        return new float[]{(float) mean, (float) variance, (float) Math.sqrt(variance)}; // Return mean, variance, and std
    }


    /**
     * Retrieve the image ID corresponding to a given title.
     *
     * @param titles Array of image titles.
     * @param ids Array of image IDs.
     * @param title The title to match.
     * @return The matching image ID, or 0 if not found.
     */
    public static int getImageIDByTitle(String[] titles, int[] ids, String title) {
        for (int i = 0; i < titles.length; i++) {
            if (titles[i].equals(title)) {
                return ids[i]; // Return the corresponding ID if title matches
            }
        }
        throw new IllegalArgumentException("Title not found: " + title); // Throw exception if title not found
    }


    /**
     * Display the results as a 2D image.
     *
     * @param inputImage The input image used for display dimensions.
     * @param repetitionMap The calculated repetition map to display.
     */
    public static void displayResults2D(Utils.InputImage2D inputImage, float[] repetitionMap) {
        FloatProcessor fp1 = new FloatProcessor(inputImage.getWidth(), inputImage.getHeight(), repetitionMap);
        ImagePlus imp1 = new ImagePlus("Repetition Map", fp1);

        // Apply SReD LUT
        InputStream lutStream = Utils.class.getResourceAsStream("/luts/sred-jet.lut");
        if (lutStream == null) {
            IJ.error("Could not load SReD LUT. Using default LUT.");
        } else {
            try {
                // Load LUT file
                IndexColorModel icm = LutLoader.open(lutStream);
                byte[] r = new byte[256];
                byte[] g = new byte[256];
                byte[] b = new byte[256];
                icm.getReds(r);
                icm.getGreens(g);
                icm.getBlues(b);
                LUT lut = new LUT(8, 256, r, g, b);

                // Apply LUT to image
                imp1.getProcessor().setLut(lut);
            } catch (IOException e) {
                IJ.error("Could not load SReD LUT: " + e.getMessage());
            }
        }

        // Display results
        imp1.show();
    }


    /**
     * Display the results as a 3D image.
     *
     * @param inputImage The input image used for display dimensions.
     * @param repetitionMap The calculated repetition map to display.
     */
    public static void displayResults3D(Utils.InputImage3D inputImage, float[] repetitionMap) {

        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        ImageStack ims = new ImageStack(imageWidth, imageHeight, imageDepth);

        for(int z=0; z<imageDepth; z++){
            FloatProcessor fp = new FloatProcessor(imageWidth, imageHeight);
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    fp.setf(x, y, repetitionMap[imageWidth*imageHeight*z+y*imageWidth+x]);
                }
            }
            ims.setProcessor(fp, z+1);
        }
        ImagePlus imp = new ImagePlus("Block repetition map", ims);
        imp.setCalibration(inputImage.getCalibration());

        // Apply SReD LUT
        InputStream lutStream = Utils.class.getResourceAsStream("/luts/sred-jet.lut");
        if (lutStream == null) {
            IJ.error("Could not load SReD LUT. Using default LUT.");
        } else {
            try {
                // Load LUT file
                IndexColorModel icm = LutLoader.open(lutStream);
                byte[] r = new byte[256];
                byte[] g = new byte[256];
                byte[] b = new byte[256];
                icm.getReds(r);
                icm.getGreens(g);
                icm.getBlues(b);
                LUT lut = new LUT(8, 256, r, g, b);

                // Apply LUT to image
                imp.setLut(lut);
            } catch (IOException e) {
                IJ.error("Could not load SReD LUT: " + e.getMessage());
            }
        }

        // Display results
        imp.show();
    }

    // ---------------------------- //
    // ---- Quadtree functions ---- //
    // ---------------------------- //



}


