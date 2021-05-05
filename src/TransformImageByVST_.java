import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.measure.Minimizer;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.awt.*;

import static java.lang.Math.sqrt;


public class TransformImageByVST_ implements PlugIn {

    @Override
    public void run(String s) {

        // Display dialog box for user input
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Calculate VST...");
        gd.addNumericField("Gain guess (default: 0)", 0);
        gd.addNumericField("Offset guess (default: 0)", 0);
        gd.addNumericField("Noise standard-deviation guess (default: 100)", 100);
        gd.addCheckbox("Estimate offset and StdDev from ROI", true);
        gd.showDialog();

        if (gd.wasCanceled()) return;

        // Grab image
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp == null) {
            IJ.log("1...");
            IJ.error("No image open, you suck!");
            return;
        }

        // Grab variables
        double gain = gd.getNextNumber();
        double offset = gd.getNextNumber();
        double sigma = gd.getNextNumber();
        boolean useROI = gd.getNextBoolean();

        IJ.showStatus("Loading up hyperdrive...");
        IJ.log("Loading up hyperdrive...");

        // Calculate offset and sigma from user-defined ROI (if user chooses to)
        ImageProcessor ip = null;
        if (useROI) {
            Roi roi = imp.getRoi();
            if (roi == null) {
                IJ.error("No ROI selected, you suck!");
                return;
            }

            ip = imp.getProcessor();
            Rectangle rect = ip.getRoi();

            int rx = rect.x;
            int ry = rect.y;
            int rw = rect.width;
            int rh = rect.height;

            // Single pass stddev https://www.strchr.com/standard_deviation_in_one_pass

            double[] values = new double[rw * rh];
            IJ.log("rx:" + rx + " ry:" + ry + " rw:" + rw + " rh:" + rh + " w:" + ip.getWidth() + " h:" + ip.getHeight());

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

            IJ.log("Offset:" + offset + " Sigma:" + sigma);

        }


        // Apply GAT to image
        FloatProcessor ifp = imp.getProcessor().convertToFloatProcessor(); // Convert ImageProcessor to FloatProcessor because minimizer() takes floats
        float[] pixels = (float[]) ifp.getPixels(); // Get pixel array
        int width = ifp.getWidth(); // Get image width
        int height = ifp.getHeight(); // Get image height
        GATMinimizer minimizer = new GATMinimizer(pixels, width, height, gain, sigma, offset); // Run minimizer
        IJ.log("DONE");

        // Display transformed image
        FloatProcessor ifp2 = new FloatProcessor(width, height, pixels);
        ImagePlus ip2 = new ImagePlus("RESULT", ifp2);
        ip2.show();

    }


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


    }









