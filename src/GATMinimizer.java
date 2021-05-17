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

        // Get error for each 64x64 segment (my take)
        FloatProcessor ifp2 = new FloatProcessor(width, height, pixels);
        int width = ifp2.getWidth(); // image width
        int height = ifp2.getHeight(); // image height
        int stepX = 64; // bounding box width
        int stepY = 64; // bounding box height
        int nBlocks = (width/stepX)*(height/stepY); // Number of bounding boxes. "Int" already cuts-off the edges
        double mean = 0;
        double var = 0;
        double error = 0;
        ifp2.setRoi(0,0,stepX,stepY); // Create bounding box at (0,0)
        Rectangle r = ifp2.getRoi();

        for (int q=0; q<height-stepY; q++) { // for each stepY..
            for (int p = 0; p < width - stepX; p++) { // ...and for each stepX
                // ADD CONDITIONAL HERE FOR UNDERSIZED BOXES I.E. EDGES! Previous loops' constraints might be enough...
                int pos, i;
                for (int y = r.y; y < (r.y + r.height); y++) { // for each y coordinate in the box..
                    pos = y * width;
                    for (int x = r.x; x < (r.x + r.width); x++) { // ... and for each x coordinate in each y
                        i = pos + x; // get position
                        mean += pixels[i]; // add the value in that position to the "mean"
                        var += pixels[i];
                    }
                    mean /= r.width*r.height; // get mean of box (didn't use r.getSize() because it returns a Dimension object)
                    var -= r.width*r.height*mean;
                    var = pow(var,2);
                    var /= r.width*r.height;
                    double delta = var - 1;
                    error += (delta * delta) / nBlocks;
                }
                r.setLocation(stepX * p, stepY * q); // move box along X
            }
            r.setLocation(0, stepY * q); // move box along Y
        }

        /*int stepX = Math.min(3, width/100);
        int stepY = Math.min(3, height/100);
        int nBlocks = (width-stepX-1)*(height-stepY-1)/(stepX*stepY);
        double error = 0;

            for (int j = 1; j < height - stepY - 1; j+=stepY) {
                for (int i = 1; i < width - stepX - 1; i+=stepX) {
                    double mean = 0;
                    mean += pixelsGAT[(j-1) * width + (i-1)];
                    mean += pixelsGAT[(j-1) * width + (i  )];
                    mean += pixelsGAT[(j-1) * width + (i+1)];
                    mean += pixelsGAT[(j+1) * width + (i-1)];
                    mean += pixelsGAT[(j+1) * width + (i  )];
                    mean += pixelsGAT[(j+1) * width + (i+1)];
                    mean += pixelsGAT[(j  ) * width + (i-1)];
                    mean += pixelsGAT[(j  ) * width + (i  )];
                    mean += pixelsGAT[(j  ) * width + (i+1)];
                    mean /= 9;

                    double var = 0;
                    var += pow(pixelsGAT[(j-1) * width + (i-1)] - mean, 2);
                    var += pow(pixelsGAT[(j-1) * width + (i  )] - mean, 2);
                    var += pow(pixelsGAT[(j-1) * width + (i+1)] - mean, 2);
                    var += pow(pixelsGAT[(j+1) * width + (i-1)] - mean, 2);
                    var += pow(pixelsGAT[(j+1) * width + (i  )] - mean, 2);
                    var += pow(pixelsGAT[(j+1) * width + (i+1)] - mean, 2);
                    var += pow(pixelsGAT[(j  ) * width + (i-1)] - mean, 2);
                    var += pow(pixelsGAT[(j  ) * width + (i  )] - mean, 2);
                    var += pow(pixelsGAT[(j  ) * width + (i+1)] - mean, 2);
                    var /= 9;

                    // according to http://www.irisa.fr/vista/Papers/2009_TMI_Boulanger.pdf
                    // var = gain * mean + sigma2 - gain * offset
                    //double delta = var - ((gain * mean) + (sigma * sigma) - gain * offset);

                    double delta = var - 1; // variance must be ~1
                    error += (delta * delta) / nBlocks;
                    //error += pow(var/(gain*mean+sigma*sigma-gain*offset)-1, 2) / pixelsGAT.length;

            }
        }*/
        return error;

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


