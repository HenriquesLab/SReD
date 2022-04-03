import ij.IJ;
import ij.measure.Minimizer;
import ij.measure.UserFunction;

import static java.lang.Math.min;
import static java.lang.Math.sqrt;


public class GATMinimizer implements UserFunction {

    public double sigma;
    private final int width, height;
    public double gain;
    public double offset;
    //public boolean isCalculated = false;
    public boolean showProgress = false;
    private final float[] pixels;

    public GATMinimizer(float[] pixels, int width, int height, double gain, double sigma, double offset){
        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.gain = gain;
        this.sigma = sigma;
        this.offset = offset;
    }

    public void run() {
        double[] initialParameters = new double[3]; // gain, sigma, offset
        double[] initialParametersVariation = new double[3];
        initialParameters[0] = gain;
        initialParameters[1] = sigma;
        initialParameters[2] = offset;
        initialParametersVariation[0] = 1;
        initialParametersVariation[1] = 10;
        initialParametersVariation[2] = 100;

        Minimizer minimizer = new Minimizer();
        minimizer.setFunction(this, 3);
        minimizer.setMaxError(0); // RH: lets figure out why? AM: This allows the minimizer to run until the relative error of the function is 0, in contrast with the default 1e-10.
        if (showProgress) minimizer.setStatusAndEsc("Estimating gain, sigma & offset: Iteration ", true);
        minimizer.minimize(initialParameters, initialParametersVariation);

        double[] params = minimizer.getParams();
        gain = params[0];
        sigma = params[1];
        offset = params[2];

        //gain = gain == 0? params[0]: gain;
        //sigma = sigma == 0? params[1]: sigma;
        //offset = offset == 0? params[2]: offset;
    }

    @Override
    public double userFunction(double[] params, double v) {
        double gain = params[0];
        double sigma = params[1];
        double offset = params[2];

        if (gain <= 0) return Double.NaN;
        if (sigma < 0) return Double.NaN;
        if (offset < 0) return Double.NaN;

        float[] pixelsGAT = pixels.clone();
        applyGAT(pixelsGAT, gain, sigma, offset);

        // ---- Get block dimensions (adjusted to image size to avoid out of bounds) ----
        // Get min
        int blockWidth = 64; // bounding box width
        int blockHeight = 64; // bounding box height

        if(width < 64*2 || height < 64*2) {
            blockWidth = 32;
            blockHeight = 32;
        }

        if(width < 32*2 || height < 32*2) {
            blockWidth = 16;
            blockHeight = 16;
        }

        if(width < 16*2 || height < 16*2) {
            blockWidth = 8;
            blockHeight = 8;
        }

        if(width < 8*2 || height < 8*2) {
            blockWidth = 4;
            blockHeight = 4;
        }

        if(width < 4*2 || height < 4*2) {
            blockWidth = 2;
            blockHeight = 2;
        }

        // Get number of blocks
        int nBlockX = width / blockWidth;
        int nBlockY = height / blockHeight;
        int nBlocks = nBlockX * nBlockY;

        double error = 0;

        for (int bY=0; bY<nBlockY; bY++) {
            for (int bX=0; bX<nBlockX; bX++) {

                int xStart = bX * blockWidth;
                int xEnd = (bX+1) * blockWidth;
                int yStart = bY * blockHeight;
                int yEnd = (bY+1) * blockHeight;

                double [] meanAndVar = getMeanAndVarBlock(pixelsGAT, xStart, yStart, xEnd, yEnd);
                double delta = meanAndVar[1] - 1; // variance must be ~1

                error += (delta * delta) / nBlocks;
            }
        }
        IJ.log("gain:"+gain+" sigma:"+sigma+" offset:"+offset+" error: " + error);

        return error;
    }

    // ---- USER METHODS ----
    // Get 1-D coordinates
    public int get1DCoordinate(int x, int y) {
        return y * width + x;
    }

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
                float v = pixels[get1DCoordinate(i,j)];
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


