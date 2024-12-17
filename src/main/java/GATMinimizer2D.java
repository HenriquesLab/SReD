/**
 *
 * Returns a set of optimized GAT parameters (gain, sigma and offset) resulting in a remapping of the input with a variance as close to 1 as possible
 *
 * @author Afonso Mendes
 *
 **/

import ij.IJ;
import ij.measure.Minimizer;
import ij.measure.UserFunction;
import static java.lang.Math.sqrt;


public class GATMinimizer2D implements UserFunction {
//TODO: THIS BUGS OUT WHEN IMAGE IS TOO SMAL DUE TO HOW WE DEFINE BLOCK SIZE
    public double gain, sigma, offset;
    private final int width, height, maxIter;
    //public boolean isCalculated = false;
    public boolean showProgress = false;
    private final float[] pixels;

    public GATMinimizer2D(float[] pixels, int width, int height, double gain, double sigma, double offset, int maxIter){
        this.pixels = pixels;
        this.width = width;
        this.height = height;
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
        minimizer.setMaxIterations(maxIter); // Set max iterations

        // Run the minimizer
        if (showProgress) minimizer.setStatusAndEsc("Estimating gain, sigma & offset: Iteration ", true);
        minimizer.minimize(initialParameters, initialParametersVariation);

        double[] params = minimizer.getParams();
        gain = params[0];
        sigma = params[1];
        offset = params[2];
    }

    @Override
    public double userFunction(double[] params, double v) {
        double gain = params[0];
        double sigma = params[1];
        double offset = params[2];

        //if (gain <= 0) return Double.NaN;
        //if (sigma < 0) return Double.NaN;
        //if (offset < 0) return Double.NaN;

        float[] pixelsGAT = pixels.clone();
        applyGAT(pixelsGAT, gain, sigma, offset);

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


        double error = 0;
        for (int bY=0; bY<nBlocksY; bY++) {
            for (int bX=0; bX<nBlocksX; bX++) {

                int xStart = bX * blockWidth;
                int xEnd = (bX+1) * blockWidth;
                int yStart = bY * blockHeight;
                int yEnd = (bY+1) * blockHeight;

                double [] meanAndVar = getMeanAndVarBlock(pixelsGAT, xStart, yStart, xEnd, yEnd);
                double delta = meanAndVar[1] - 1; // variance must be ~1

                error += (delta * delta) / (double)nBlocks;
            }
        }
        //IJ.log("gain:"+gain+" sigma:"+sigma+" offset:"+offset+" error: " + error);
        return error;
    }

    // ---- USER METHODS ----

    // Get mean and variance of a patch
    public double[] getMeanAndVarBlock(float[] pixels, int xStart, int yStart, int xEnd, int yEnd) {
        double mean = 0;
        double var = 0;

        double sq_sum = 0;

        int bWidth = xEnd-xStart;
        int bHeight = yEnd - yStart;
        int bWH = bWidth*bHeight;

        for (int j=yStart; j<yEnd; j++) {
            for (int i=xStart; i<xEnd; i++) {
                float v = pixels[j*width+i];
                mean += v;
                sq_sum += v * v;
            }
        }

        mean = mean / bWH;
        var = sq_sum / bWH - mean * mean;

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


