import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

import java.awt.*;

import static java.lang.Math.sqrt;


public class TransformImageByVST_ implements PlugIn {

    @Override
    public void run(String s) {

        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Calculate VST...");
        gd.addNumericField("Gain guess (default: 0)", 0);
        gd.addNumericField("Offset guess (default: 0)", 0);
        gd.addNumericField("Noise standard-deviation guess (default: 100)", 100);
        gd.addCheckbox("Estimate offset and StdDev from ROI", true);
        gd.showDialog();

        if (gd.wasCanceled()) return;

        // Grab image
        ImagePlus imp = WindowManager.getCurrentImage();
        if (imp==null) {
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

        if (useROI) {
            Roi roi = imp.getRoi();
            if (roi == null) {
                IJ.error("No ROI selected, you suck!");
                return;
            }

            ImageProcessor ip = imp.getProcessor();
            Rectangle rect = ip.getRoi();

            int rx = rect.x;
            int ry = rect.y;
            int rw = rect.width;
            int rh = rect.height;

            // Single pass stddev https://www.strchr.com/standard_deviation_in_one_pass

            double[] values = new double[rw*rh];
            IJ.log("rx:"+rx+" ry:"+ry+" rw:"+rw+" rh:"+rh+" w:"+ip.getWidth()+" h:"+ip.getHeight());

            int counter=0;
            for (int j=ry;j<ry+rh;j++) {
                for (int i=rx;i<rx+rw;i++) {
                    values[counter]=ip.getPixel(i,j);
                    counter++;
                }
            }
            double[] offsetAndSigma = meanAndStdDev(values);
            offset = offsetAndSigma[0];
            sigma = offsetAndSigma[1];

            IJ.log("Offset:"+offset+" Sigma:"+sigma);
        }

        // Get dark current error, eDC (sigma^2-gain*offset)
        double edc = Math.pow(sigma,2)-gain*offset;
        IJ.log("Dark current error (eDC):"+edc);
        
    }

    private double[] meanAndStdDev(double a[]) {
        int n = a.length;
        if (n == 0) return new double[]{0,0};

        double sum = 0;
        double sq_sum = 0;

        for (int i = 0; i < n; i++) {
            sum += a[i];
            sq_sum += a[i] * a[i];
        }

        double mean = sum / n;
        double variance = sq_sum / n - mean * mean;

        return new double[] {mean, sqrt(variance)};
    }
}
