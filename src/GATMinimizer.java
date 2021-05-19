import ij.IJ;
import ij.measure.Minimizer;
import ij.measure.UserFunction;
import ij.process.FloatProcessor;

import java.awt.*;
import java.util.ArrayList;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;


public class GATMinimizer implements UserFunction {

    public double sigma;
    private final int width, height, widthHeight;
    public double gain;
    public double offset;
    //public boolean isCalculated = false;
    public boolean showProgress = false;
    private final float[] pixels;

    public GATMinimizer(float[] pixels, int width, int height, double gain, double sigma, double offset){
        this.pixels = pixels;
        this.width = width;
        this.height = height;
        this.widthHeight = width*height;
        this.gain = gain;
        this.sigma = sigma;
        this.offset = offset;
    }

    public void run() {
        double[] initialParameters = new double[3]; // gain, sigma, offset
        double[] initialParametersVariation = new double[3];
        initialParameters[0] = 1;
        initialParameters[1] = 0;
        initialParameters[2] = 0;
        initialParametersVariation[0] = 1;
        initialParametersVariation[1] = 10;
        initialParametersVariation[2] = 100;

        Minimizer min = new Minimizer();
        min.setFunction(this, 3);
        min.setMaxError(0); // lets figure out why?
        if (showProgress) min.setStatusAndEsc("Estimating gain, sigma & offset: Iteration ", true);
        min.minimize(initialParameters, initialParametersVariation);

        double[] params = min.getParams();
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

        System.out.println(pixels);

        float[] pixelsGAT = pixels.clone();
        applyGeneralizedAnscombeTransform(pixelsGAT, gain, sigma, offset);

        int blockWidth = 64; // bounding box width
        int blockHeight = 64; // bounding box height

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

    public int get1DCoordinate(int x, int y) {
        return y * width + x;
    }

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

    public static void applyGeneralizedAnscombeTransform(float[] pixels, double gain, double sigma, double offset) {

        double refConstant = (3d/8d) * gain * gain + sigma * sigma - gain * offset;
        //its called refConstant because it does not contain the pixel values, Afonso was confused and needed a hug

        // Apply GAT to pixel value
        // See http://mirlab.org/conference_papers/International_Conference/ICASSP%202012/pdfs/0001081.pdf for GAT description

        for (int n=0; n<pixels.length; n++) {
            double v = pixels[n];
            if (v <= -refConstant/gain) v = 0; // checking for a special case, Ricardo does not remember why, he's 40 after all
            else v = (2 / gain) * sqrt(gain * v + refConstant);
            pixels[n] = (float) v;
        }
        System.out.println(pixels);
    }

}


