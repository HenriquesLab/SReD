/**
 *
 * Returns a set of optimized GAT parameters (gain, sigma and offset) resulting in a remapping of the input with a variance as close to 1 as possible.
 * Importantly, the input is expected to be a 3D image and the parameters are optimized to the variance of all slices.
 * Each slice is treated indepedently but the best set of parameters is the one that results in a variance as close to 1 as possible across all slices.
 *
 * @author Afonso Mendes
 *
 **/

import ij.IJ;
import ij.measure.Minimizer;
import ij.measure.UserFunction;

import static java.lang.Math.max;
import static java.lang.Math.sqrt;


public class GATMinimizer3D implements UserFunction {

    public double sigma;
    private final int width, height, depth, maxIter;
    public double gain;
    public double offset;
    //public boolean isCalculated = false;
    public boolean showProgress = true;
    private final float[] pixels;

    public GATMinimizer3D(float[] pixels, int width, int height, int depth, double gain, double sigma, double offset, int maxIter){
        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.gain = gain;
        this.sigma = sigma;
        this.offset = offset;
        this.maxIter = maxIter;
    }

    public void run() {

        // Define initial parameters
        double[] initialParameters = new double[3]; // gain, sigma, offset
        double[] initialParametersVariation = new double[3];
        initialParameters[0] = gain;
        initialParameters[1] = sigma;
        initialParameters[2] = offset;

        // Define initial parameters variation
        initialParametersVariation[0] = 1;
        initialParametersVariation[1] = 10;
        initialParametersVariation[2] = 100;

        // Create the minimizer
        Minimizer minimizer = new Minimizer();
        minimizer.setFunction(this, 3);
        minimizer.setMaxError(0); // Allows the minimizer to run until the relative error of the function is 0, in contrast with the default 1e-10.
        minimizer.setMaxIterations(maxIter);

        // Run the minimizer
        if (showProgress) minimizer.setStatusAndEsc("Estimating GAT parameters: Iteration ", true);
        minimizer.minimize(initialParameters, initialParametersVariation);

        // Get the optimized parameters
        double[] params = minimizer.getParams();
        gain = params[0];
        sigma = params[1];
        offset = params[2];
    }

    @Override

    // --------------------------------------- //
    // ---- Define the objective function ---- //
    // --------------------------------------- //
    public double userFunction(double[] params, double v) {

        // Get optimized parameters
        double gain = params[0];
        double sigma = params[1];
        double offset = params[2];

        // Check if parameters are valid
        if (gain <= 0) return Double.NaN;
        if (sigma < 0) return Double.NaN;
        if (offset < 0) return Double.NaN;

        // Get copy of input image to transform
        //float[][] pixelsGAT = new float[depth][width * height];
        //for (int z = 0; z < depth; z++){
        //    pixelsGAT[z] = pixels[z].clone();
        //}

        int whz = width*height*depth; // Size of the data
        float[] pixelsGAT = new float[whz];
        for(int i=0; i<whz; i++){
            pixelsGAT[i] = pixels[i];
        }

        // Apply GAT using current parameters
        GATMinimizer2D.applyGAT(pixelsGAT, gain, sigma, offset);

        // Define the dimensions of the window used to estimate the noise variance
        // max() is used to prevent block dimensions == 0
        int blockWidth = max(width/6, 1); // bounding box width
        int blockHeight = max(height/6, 1); // bounding box height
        int blockDepth = max(depth/6, 1); // Bounding box depth

        // Get number of blocks
        int nBlocksX = width / blockWidth;
        int nBlocksY = height / blockHeight;
        int nBlocksZ = depth / blockDepth;

        // Ensure that you have at least 2 blocks in XYZ
        while(nBlocksX<2 || nBlocksY<2 || nBlocksZ<2) {
            if (nBlocksX < 2 && blockWidth > 1) blockWidth = Math.max(blockWidth / 2, 1);
            if (nBlocksY < 2 && blockHeight > 1) blockHeight = Math.max(blockHeight / 2, 1);
            if (nBlocksZ < 2 && blockDepth > 1) blockDepth = Math.max(blockDepth / 2, 1);

            // Recalculate number of blocks
            nBlocksX = width / blockWidth;
            nBlocksY = height / blockHeight;
            nBlocksZ = depth / blockDepth;

            // If further halving would result in invalid blocks, break the loop
            if (blockWidth == 1 && blockHeight == 1 && blockDepth == 1) break;
        }

        int nBlocks = nBlocksX * nBlocksY * nBlocksZ;

        // Evaluate the noise variance error by calculating the variance of each block and its difference from 1
        double error = 0.0d;

        for(int bZ=0; bZ<nBlocksZ; bZ++) {
            for (int bY=0; bY<nBlocksY; bY++) {
                for (int bX=0; bX<nBlocksX; bX++) {

                    int xStart = bX * blockWidth;
                    int xEnd = (bX + 1) * blockWidth;

                    int yStart = bY * blockHeight;
                    int yEnd = (bY + 1) * blockHeight;

                    int zStart = bZ * blockDepth;
                    int zEnd = (bZ + 1) * blockDepth;

                    double variance = Utils.getMeanAndVarBlock3D(pixelsGAT, width, height, xStart, yStart, zStart, xEnd, yEnd, zEnd)[1];
                    double delta = variance - 1.0d; // variance must be ~1

                    error += (delta * delta) / (double)nBlocks;
                }
            }
        }
        //IJ.log("gain:"+gain+" sigma:"+sigma+" offset:"+offset+" error: " + error);
        return error;
    }


}


