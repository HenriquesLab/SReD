/**
 * Created by Afonso Mendes and Ricardo Henriques on April 2021.
 *
 * Calls the "GATMinimizer2D" class to calculate the gain, sigma and offset of a 2D image that result in a variance as close as possible to 1.
 * Remaps pixels to stabilize the noise variance using the Generalized Anscombe Transform using the optimized set of parameters.
 *
 * @author Afonso Mendes
 * @author Ricardo Henriques
 *
 */

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.*;
import static java.lang.Math.sqrt;


public class VarianceStabilisingTransform2D_ implements PlugIn {

    @Override
    public void run(String s) {

        // ---- Display dialog box for user input ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Calculate VST...");
        gd.addNumericField("Gain guess:", 0);
        gd.addNumericField("Offset guess:", 0);
        gd.addNumericField("Noise standard deviation guess:", 100);
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


        // ---- Apply GAT to image ----
        // Grab image and get pixel values dimensions
        FloatProcessor ifp = imp.getProcessor().convertToFloatProcessor(); // Convert to FloatProcessor because minimizer() requires floats
        float[] pixels = (float[]) ifp.getPixels(); // Get pixel array
        int width = ifp.getWidth(); // Get image width
        int height = ifp.getHeight(); // Get image height

        // Run the optimizer to find gain, offset and sigma that minimize the noise variance
        GATMinimizer2D minimizer = new GATMinimizer2D(pixels, width, height, gain, sigma, offset); // Run minimizer
        minimizer.run();

        // Create final "variance stable" image based on optimized parameters
        float[] pixelsGAT;
        pixelsGAT = getGAT(pixels, minimizer.gain, minimizer.sigma, minimizer.offset);
        FloatProcessor fp1 = new FloatProcessor(width, height, pixelsGAT);
        ImagePlus imp1 = new ImagePlus("Variance-stabilized image", fp1);
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
                v = 0; // checking for a special case, Ricardo does not remember why, he's 40 after all. AM: 40/10!
            }else {
                v = (2 / gain) * sqrt(gain * v + refConstant);
            }

            pixels[n] = (float) v;
        }
        return pixels;
    }
}









