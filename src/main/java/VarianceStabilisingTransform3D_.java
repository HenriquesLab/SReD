/**
 * Created by Afonso Mendes and Ricardo Henriques on April 2021.
 *
 * Calls the "GATMinimizer3D" class to calculate the gain, sigma and offset of a 3D image that result in a variance as close as possible to 1.
 * Remaps pixels to stabilize the noise variance using the Generalized Anscombe Transform using the optimized set of parameters.
 *
 * @author Afonso Mendes
 * @author Ricardo Henriques
 *
 */

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;


public class VarianceStabilisingTransform3D_ implements PlugIn {

    @Override
    public void run(String s) {

        // ---- Display dialog box for user input ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Noise variance stabilisation 3D (Simplex)");
        gd.addMessage("Initial parameter value guesses:");
        gd.addNumericField("Gain:", 1);
        gd.addNumericField("Offset:", 10);
        gd.addNumericField("Noise StdDev:", 100);
        gd.addMessage("");
        gd.addNumericField("Max iterations:", 5000);
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
        int maxIter = (int) gd.getNextNumber();

        // Grab image stack and its dimensions
        ImageStack ims = imp.getImageStack();
        int width = ims.getWidth(); // Image width
        int height = ims.getHeight(); // Image height
        int depth = ims.getSize(); // Image depth
        int size = width*height*depth; // Image size

        float[] pixels = new float[size];
        for (int z=0; z<depth; z++) {
            FloatProcessor fp = ims.getProcessor(z+1).convertToFloatProcessor();
            for(int y=0; y<height; y++) {
                for(int x=0; x<width; x++) {
                    pixels[width*height*z+y*width+x] = fp.getf(x,y);
                }
            }
        }

        // Run an optimizer to find gain, offset and sigma that minimize the noise variance
        IJ.log("Stabilising noise variance (Simplex method)...");
        GATMinimizer3D minimizer = new GATMinimizer3D(pixels, width, height, depth, gain, sigma, offset, maxIter); // Run minimizer
        minimizer.run();

        // Apply GAT to image using optimized parameter values
        float[] pixelsGAT;
        pixelsGAT = VarianceStabilisingTransform2D_.getGAT(pixels, minimizer.gain, minimizer.sigma, minimizer.offset);

        // Build and show output
        ImageStack ims1 = new ImageStack(width, height, depth); // Final ImageStack
        for (int z=0; z<depth; z++) {
            FloatProcessor fp = new FloatProcessor(width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    fp.setf(x, y, pixelsGAT[width*height*z+y*width+x]);
                }
            }
            ims1.setProcessor(fp, z+1);
        }
        ImagePlus imp1 = new ImagePlus("Variance-stabilized image", ims1);
        imp1.show();

        // Print estimated parameter values
        // Print optimized parameter values
        IJ.log("Gain = " + minimizer.gain);
        IJ.log("Sigma = " + minimizer.sigma);
        IJ.log("Offset = " + minimizer.offset);
        IJ.log("Done!");    }
}









