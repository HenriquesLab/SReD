/**
 *
 * TODO: Implement progress tracking
 * TODO: Solve redundancy map having an extra column
 **/

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import static java.lang.Math.*;

public class RedundancyMap_ implements PlugIn {
    @Override
    public void run(String s) {

        // ---- Get reference image and some parameters ----
        ImagePlus imp0 = WindowManager.getCurrentImage();
        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = imp0.getWidth();
        int h = imp0.getHeight();
        float sigma = 1.7F; // TODO: This should be the noise STDDEV, which can be taken from a dark patch in the image

        // ---- Create array to store Redundancy Map pixels ----
        float[] finalPixels = new float[w * h];

        // ---- Patch parameters ----
        int bW = 3; // Width
        int bH = 3; // Height

        // ---- SLIDING WINDOW OPERATIONS ----
        IJ.log("Calculating redundancy map...");
        // Loop through reference pixels
        float[] refVar = new float[finalPixels.length]; // DELETE
        float[] refMean = new float[finalPixels.length]; // DELETE

        for (int y = 1; y < h - 2; y++) {
            for (int x = 1; x < w - 2; x++) {

                ComparisonThread comparison = new ComparisonThread();
                comparison.setup(refPixels, finalPixels, w, h, bW, bH, x, y, sigma, refVar, refMean);
                comparison.start();
            }
        }

        //---- Create redundancy map and display it ----
        FloatProcessor fp1 = new FloatProcessor(w, h, finalPixels);
        ImagePlus imp1 = new ImagePlus("Redundancy Map", fp1);
        imp1.show();

        IJ.log("Done!");

    }
}

class ComparisonThread extends Thread {

    private float[] refPixels;
    private float[] finalPixels;
    private int x, y, w, h, bW, bH;
    private float sigma;

    public void setup(float[] refPixels, float[] finalPixels, int w, int h, int bW, int bH, int x, int y, float sigma, float[] refVar, float[] refMean) {
        this.refPixels = refPixels;
        this.finalPixels = finalPixels;
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.bW = bW;
        this.bH = bH;
        this.sigma = sigma;
    }

    @Override
    public void run() {

        float[] currentPearsonList = new float[refPixels.length];
        float[] currentWeightList = new float[refPixels.length];

        // Get current reference patch pixels
        float pixels0[] = new float[bW*bH];
        int bRW = bW/2;
        int bRH = bH/2;


        // Fill reference patch
        int refCounter = 0;
        for (int j=y-bRH; j<y+bRH;j++) {
            for (int i=x-bRW; i<x+bRW;i++) {
                pixels0[refCounter] = this.refPixels[j * w + i];
                refCounter++;
            }
        }

        // Get statistics for the current reference patch
        float[] patchStats0 = getPatchStats(pixels0);

        // Get array of mean-subtracted pixels for the current reference patch
        float[] pixels0meansub = new float[bW*bH];
        for (int b = 0; b < bW*bH; b++) {
            pixels0meansub[b] = pixels0[b]-patchStats0[0];
        }

        // Loop through comparison pixels
        for (int y1 = 1; y1 < h - 2; y1++) {
            for (int x1 = 1; x1 < w - 2; x1++) {

                // Fill comparison patch
                float pixels1[] = new float[bW*bH];

                int compCounter = 0;
                for (int j1=y1-bRH; j1<y1+bRH;j1++) {
                    for (int i1=x1-bRW; i1<x1+bRW;i1++) {
                        pixels1[compCounter] = this.refPixels[j1 * w + i1];
                        compCounter++;
                    }
                }

                // Get mean and standard deviation of current comparison patch
                float[] patchStats1 = getPatchStats(pixels1);

                // Compare standard deviation between patches to decide if it's worth doing patch-wise comparison
                float preComparison = abs(patchStats0[1]-patchStats1[1]);
                float pearson;
                float weight;

                // Mean-subtract each pixel in the current comparison patch
                float[] pixels1meansub = new float[bW*bH];
                for (int d = 0; d < bW*bH; d++) {
                    pixels1meansub[d] = pixels1[d]-patchStats1[0];
                }

                if (preComparison <= patchStats0[1]*2) {
                    // Get Pearson correlation coefficient, truncate, and get this Pearson's weight
                    float num = 0;

                    for (int d = 0; d < bW*bH; d++) {
                        num += pixels0meansub[d]*pixels1meansub[d];
                    }

                    pearson = (float) max(0, num/sqrt(patchStats0[1])*sqrt(patchStats1[1])); // Truncated value TODO: CHECK DIVISION BY ZERO

                    weight = (float) exp((-1)*pow(abs(patchStats1[0]-patchStats0[0]),2)/pow(0.4F*sigma,2)); // Non-local means Gaussian weight function; https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions TODO:Check division by zero
                    //weight = 1-abs(patchStats0[0]-patchStats1[0]/abs(patchStats0[0]+abs(patchStats1[0])));

                } else {
                    // Store an arbitrary Pearson correlation coefficient
                    pearson = 0; //TODO: Here I chose 0 as a representation of the lowest possible Pearson correlation. Decide later on what to do here
                    weight = 0;

                }

                // Store Pearson's correlation coefficient and its corresponding weight
                currentPearsonList[y1*w+x1] = pearson;
                currentWeightList[y1*w+x1] = weight;

            }
        }

        // Paint the Redundancy Map. Each pixel is a weighted mean of the current Pearson's correlation coefficients
        finalPixels[y*w+x] = getArrayWeightedMean(currentPearsonList,currentWeightList);
    }

    // ---- USER METHODS ----

    // Get patch statistics (single pass, see https://www.strchr.com/standard_deviation_in_one_pass)
    private float[] getPatchStats(float a[]) {
        int n = a.length;
        if (n == 0) return new float[]{0, 0, 0, 0};

        float sum = 0;
        float sq_sum = 0;

        for (int i = 0; i < n; i++) {
            sum += a[i];
            sq_sum += a[i] * a[i];
        }

        float mean = max(sum/n, 1); // Values below 1 are stored as 1 to avoid divisions by zero and decimals
        float variance = sq_sum / n - mean * mean;

        return new float[]{mean, (float) sqrt(variance), variance, sum};
    }

    // Get array mean
    private float getArrayMean(float a[]) {
        float arrayMean = 0;

        for (int i=0; i<a.length;i++){
            arrayMean += a[i];
        }

        arrayMean /= a.length;
        return max(arrayMean,1); // Values below 1 are stored as 1 to avoid divisions by zero and decimals
    }

    // Get array weighted mean
    private float getArrayWeightedMean(float a[], float b[]) {
        float weightedMean = 0;

        for (int i=0; i<a.length;i++){
            weightedMean += a[i]*b[i];
        }

        weightedMean /= a.length;
        return weightedMean;
    }

}
