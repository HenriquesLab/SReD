/**
 * Created by Afonso Mendes and Ricardo Henriques on April 2021.
 * TODO: Solve issue with VST final image being weird when parameters are estimated from ROI
 */

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.awt.*;
import static java.lang.Math.sqrt;


public class VarianceStabilisingTransform3D_ implements PlugIn {

    @Override
    public void run(String s) {

        // ---- Display dialog box for user input ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Calculate VST...");
        gd.addNumericField("Gain guess:", 0);
        gd.addNumericField("Offset guess:", 0);
        gd.addNumericField("Noise standard-deviation guess:", 100);
        gd.addCheckbox("Estimate offset and StdDev from ROI?", false);
        gd.showDialog();

        if (gd.wasCanceled()) return;

        // ---- Define gain, offset and sigma ----
        // Grab image
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        if (imp.getNSlices() < 2) {
            IJ.error("Image is not 3D. Please open a 3D image and try again.");
            return;
        }

        // Grab variables from dialog box
        double gain = gd.getNextNumber();
        double offset = gd.getNextNumber();
        double sigma = gd.getNextNumber();
        boolean useROI = gd.getNextBoolean();

        // Calculate offset and sigma from user-defined ROI (if user chooses to)
        ImageProcessor ip = null;
        if (useROI) {
            Roi roi = imp.getRoi();
            if (roi == null) {
                IJ.error("No ROI selected. Please draw a rectangle and try again.");
                return;
            }

            ip = imp.getProcessor();
            Rectangle rect = ip.getRoi();

            int rx = rect.x;
            int ry = rect.y;
            int rw = rect.width;
            int rh = rect.height;

            // Get standard deviation (single pass) https://www.strchr.com/standard_deviation_in_one_pass
            double[] values = new double[rw * rh];

            int counter = 0;
            for (int j = ry; j < ry + rh; j++) {
                for (int i = rx; i < rx + rw; i++) {
                    values[counter] = ip.getPixel(i, j);
                    counter++;
                }
            }

            double[] offsetAndSigma = meanAndStdDev(values);
            offset = offsetAndSigma[0];
            sigma = offsetAndSigma[1];
        }


        // --------------------------------------------- //
        // ---- Grab image stack and its dimensions ---- //
        // --------------------------------------------- //

        ImageStack ims = imp.getImageStack();

        int w = ims.getWidth(); // Get image width
        int h = ims.getHeight(); // Get image height
        int z = ims.getSize(); // Get image depth

        float[][] pixels = new float[z][w*h];
        for (int i=0; i<z; i++) {
            for(int y=0; y<h; y++) {
                for(int x=0; x<w; x++) {
                    pixels[i][y*w+x] = ims.getProcessor(i+1).convertToFloatProcessor().getf(x,y);
                }
            }
        }

        // ------------------------------------------------------------------------------------------ //
        // ---- Run an optimizer to find gain, offset and sigma that minimize the noise variance ---- //
        // ------------------------------------------------------------------------------------------ //

        GATMinimizer3D minimizer = new GATMinimizer3D(pixels, w, h, z, gain, sigma, offset); // Run minimizer
        minimizer.run();


        // ------------------------------------------------------------- //
        // ---- Apply GAT to image using optimized parameter values ---- //
        // ------------------------------------------------------------- //

        float[][] pixelsGAT = new float[z][w*h];
        ImageStack ims1 = new ImageStack(w, h, z);

        for (int i=0; i<z; i++) {
            pixelsGAT[i] = getGAT(pixels[i], minimizer.gain, minimizer.sigma, minimizer.offset);
            FloatProcessor temp = new FloatProcessor(w, h);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    temp.setf(x, y, pixelsGAT[i][y * w + x]);
                }
            }
            ims1.setProcessor(temp, i+1);

        }


        // --------------------------------- //
        // ---- Display the image stack ---- //
        // --------------------------------- //

        ImagePlus imp1 = new ImagePlus("Variance-stabilized image", ims1);
        imp1.show();
    }

    // ---- USER METHODS ----
    private double[] meanAndStdDev ( double a[]){
        int n = a.length;
        if (n == 0) return new double[]{0, 0};

        double sum = 0;
        double sq_sum = 0;

        for (int i = 0; i < n; i++) {
            sum += a[i];
            sq_sum += a[i] * a[i];
        }

        double mean = sum / n;
        double variance = sq_sum / n - mean * mean;

        return new double[]{mean, sqrt(variance)};

    }

    // Get GAT (see http://mirlab.org/conference_papers/International_Conference/ICASSP%202012/pdfs/0001081.pdf for GAT description)
    public static float[] getGAT(float[] pixels, double gain, double sigma, double offset) {

        double refConstant = (3d/8d) * gain * gain + sigma * sigma - gain * offset;

        for (int n=0; n<pixels.length; n++) {
            double v = pixels[n];
            if (v <= -refConstant / gain) {
                v = 0.0; // checking for a special case, Ricardo does not remember why, he's 40 after all. AM: 40 out of 10! AM 3 yearrz laterr: it avoids the sqrt of a negative number in the else statement
            }else {
                v = (2.0 / gain) * sqrt(gain * v + refConstant);
            }

            pixels[n] = (float) v;
        }
        return pixels;
    }
}









