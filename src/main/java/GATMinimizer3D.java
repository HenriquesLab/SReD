/**
 *
 * Returns a set of optimized GAT parameters (gain, sigma and offset) resulting in a remapping of the input with a variance as close to 1 as possible.
 * Importantly, the input is expected to be a 3D image and the parameters are optimized to the variance of all slices.
 * Each slice is treated indepedently but the best set of parameters is the one that results in a variance as close to 1 as possible across all slices.
 *
 * @author Afonso Mendes
 *
 **/

import ij.measure.Minimizer;
import ij.measure.UserFunction;
import static java.lang.Math.sqrt;


public class GATMinimizer3D implements UserFunction {

    public double sigma;
    private final int width, height, depth;
    public double gain;
    public double offset;
    //public boolean isCalculated = false;
    public boolean showProgress = true;
    private final float[][] pixels;

    public GATMinimizer3D(float[][] pixels, int width, int height, int depth, double gain, double sigma, double offset){
        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.gain = gain;
        this.sigma = sigma;
        this.offset = offset;
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

        // Run the minimizer
        if (showProgress) minimizer.setStatusAndEsc("Estimating gain, sigma & offset: Iteration ", true);
        minimizer.minimize(initialParameters, initialParametersVariation);

        // Get the optimized parameters
        double[] params = minimizer.getParams();
        gain = params[0];
        sigma = params[1];
        offset = params[2];
        //gain = gain == 0? params[0]: gain;
        //sigma = sigma == 0? params[1]: sigma;
        //offset = offset == 0? params[2]: offset;
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
        float[][] pixelsGAT = new float[depth][width * height];
        for (int z = 0; z < depth; z++){
            pixelsGAT[z] = pixels[z].clone();
        }

        // Apply GAT using current parameters
        for(int z=0; z<depth; z++) {
            applyGAT(pixelsGAT[z], gain, sigma, offset);
        }

        // Define the dimensions of the window used to estimate the noise variance (adjusted to image size to avoid out of bounds)
        int blockWidth = width/6; // bounding box width
        int blockHeight = height/6; // bounding box height

        // Get number of blocks
        int nBlocksX = width / blockWidth;
        int nBlocksY = height / blockHeight;

        // Ensure that you have at least 2 blocks in XY
        while(nBlocksX<2 || nBlocksY<2) {
            blockWidth /= 2;
            blockHeight /= 2;
        }

        int nBlocks = nBlocksX * nBlocksY;


        // Evaluate the noise variance error by calculating the variance of each block and its difference from 1
        double error = 0;

        for(int z=0; z<depth; z++) {
            for (int bY=0; bY<nBlocksY; bY++) {
                for (int bX=0; bX<nBlocksX; bX++) {

                    int xStart = bX * blockWidth;
                    int xEnd = (bX + 1) * blockWidth;

                    int yStart = bY * blockHeight;
                    int yEnd = (bY + 1) * blockHeight;

                    double[] meanAndVar = getMeanAndVarBlock(pixelsGAT[z], xStart, yStart, xEnd, yEnd);
                    double delta = meanAndVar[1] - 1; // variance must be ~1

                    error += (delta * delta) / nBlocks;
                }
            }
        }
        //IJ.log("gain:"+gain+" sigma:"+sigma+" offset:"+offset+" error: " + error);
        return error;
    }

    // ---- USER METHODS ----

    // Get mean and variance of a block
    public double[] getMeanAndVarBlock(float[] pixels, int xStart, int yStart, int xEnd, int yEnd) {
        double mean = 0;
        double var;

        double sq_sum = 0;

        int nPixelsX = xEnd-xStart;
        int nPixelsY = yEnd-yStart;
        int nPixels = nPixelsX * nPixelsY;

        for (int y=yStart; y<yEnd; y++) {
            for (int x=xStart; x<xEnd; x++) {
                float v = pixels[y*width+x];
                mean += v;
                sq_sum += v*v;
            }
        }


        mean = mean / nPixels;
        var = sq_sum / nPixels - mean * mean;

        return new double[] {mean, var};
    }

    // Apply Generalized Anscombe Transform
    public static void applyGAT(float[] pixels, double gain, double sigma, double offset) {

        double refConstant = (3d/8d) * gain * gain + sigma * sigma - gain * offset;
        //it's called refConstant because it does not contain the pixel values, Afonso was confused and needed a hug

        // Apply GAT to pixel value (see http://mirlab.org/conference_papers/International_Conference/ICASSP%202012/pdfs/0001081.pdf for GAT description)
        for (int n=0; n<pixels.length; n++) {
            double v = pixels[n];
            if (v <= -refConstant / gain) {
                v = 0; // checking for a special case, Ricardo does not remember why, he's 40 after all. AM: 40 out of 10!
            }else{
                v = (2 / gain) * sqrt(gain * v + refConstant);
            }
            pixels[n] = (float) v;
        }
    }
}


