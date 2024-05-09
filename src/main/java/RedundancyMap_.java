/**
 *
 * This class acts as a master Global Repetition handler.
 * It instantiates rounds of Global Repetition calculations according to the users' needs.
 * This allows extending Global Repetition for timelapses and across scales sequentially.
 *
 * The OpenCL implementation (initialization, context creation, device selection, etc...) is also handled here.
 * OpenCL buffer, programs, kernels, etc. are handled in the "GlobalRedundancy" class and recycled for each iteration.
 * This allows using a single OpenCL context for all iterations.
 *
 * The calculation of the Global Repetition Map is handled in the "GlobalRedundancy" class.
 *
 * @author Afonso Mendes
 *
 **/

import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import java.util.ArrayList;
import java.util.List;

public class RedundancyMap_ implements PlugIn {

    // --------------------------- //
    // ---- Initialize OpenCL ---- //
    // --------------------------- //

    static private CLContext context;
    static private CLPlatform clPlatformMaxFlop;
    static private CLCommandQueue queue;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Define metric possibilities
        String[] metrics = new String[4];
        metrics[0] = "Pearson's R";
        metrics[1] = "Cosine similarity";
        metrics[2] = "Hu moments";
        metrics[3] = "mSSIM";

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Global Redundancy");
        gd.addNumericField("Block width (px): ", 3, 2);
        gd.addNumericField("Box height (px): ", 3, 2);
        gd.addCheckbox("Time-lapse?", false);
        gd.addCheckbox("Multi-scale?", false);

        gd.addSlider("Filter constant: ", 0.0f, 10.0f, 0.0, 0.1f);
        gd.addChoice("Metric:", metrics, metrics[1]);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get dialog parameters
        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = (int) gd.getNextNumber(); // Patch height

        int useTime = 0;
        if(gd.getNextBoolean() == true) {
            useTime = 1;
        }

        int multiScale = 0; // Multi-scale analysis?
        if(gd.getNextBoolean() == true){
            multiScale = 1;
        }

        //TODO: SOLVE THIS
        if(useTime == 1 && multiScale == 1) {
            IJ.error("Multi-scale analysis in timelapse data is not yet supported.");
            return;
        }

        float filterConstant = (float) gd.getNextNumber();

        String metric = gd.getNextChoice();

        boolean useDevice = gd.getNextBoolean();

        // Check if patch dimensions are odd, otherwise kill program
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        // Calculate block radius
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)

        // Get final patch size (after removing pixels outside inbound circle/ellipse)
        int patchSize = 0;
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                float dx = (float)(i-bRW);
                float dy = (float)(j-bRH);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                    patchSize++;
                }
            }
        }

        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        long start = System.currentTimeMillis();


        // --------------------------- //
        // ---- Initialize OpenCL ---- //
        // --------------------------- //

        CLPlatform[] allPlatforms = CLPlatform.listCLPlatforms();

        try {
            allPlatforms = CLPlatform.listCLPlatforms();
        } catch (CLException ex) {
            IJ.log("Something went wrong while initialising OpenCL.");
            throw new RuntimeException("Something went wrong while initialising OpenCL.");
        }

        double nFlops = 0;

        for (CLPlatform allPlatform : allPlatforms) {
            CLDevice[] allCLdeviceOnThisPlatform = allPlatform.listCLDevices();

            for (CLDevice clDevice : allCLdeviceOnThisPlatform) {
                //IJ.log("--------");
                //IJ.log("Device name: " + clDevice.getName());
                //IJ.log("Device type: " + clDevice.getType());
                //IJ.log("Max clock: " + clDevice.getMaxClockFrequency() + " MHz");
                //IJ.log("Number of compute units: " + clDevice.getMaxComputeUnits());
                //IJ.log("Max work group size: " + clDevice.getMaxWorkGroupSize());
                if (clDevice.getMaxComputeUnits() * clDevice.getMaxClockFrequency() > nFlops) {
                    nFlops = clDevice.getMaxComputeUnits() * clDevice.getMaxClockFrequency();
                    clPlatformMaxFlop = allPlatform;
                }
            }
        }
        IJ.log("--------");

        // Create context
        context = CLContext.create(clPlatformMaxFlop);

        // Choose the best device (i.e., Filter out CPUs if GPUs are available (FLOPS calculation was giving
        // higher ratings to CPU vs. GPU))
        //TODO: Rate devices on Mandelbrot and chooose the best, GPU will perform better
        CLDevice[] allDevices = context.getDevices();

        boolean hasGPU = false;
        for (int i = 0; i < allDevices.length; i++) {
            if (allDevices[i].getType() == CLDevice.Type.GPU) {
                hasGPU = true;
            }
        }
        CLDevice chosenDevice;
        if (hasGPU) {
            chosenDevice = context.getMaxFlopsDevice(CLDevice.Type.GPU);
        } else {
            chosenDevice = context.getMaxFlopsDevice();
        }

        // Get chosen device from preferences
        if(useDevice){
            String deviceName = Prefs.get("SReD.OpenCL.device", null);
            for (CLDevice device : allDevices) {
                if (device.getName().equals(deviceName)) {
                    chosenDevice = device;
                    break;
                }
            }
        }

        IJ.log("Chosen device: " + chosenDevice.getName());
        IJ.log("--------");

        // Create command queue
        queue = chosenDevice.createCommandQueue();


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels0 = (float[]) fp0.getPixels();
        int w0 = fp0.getWidth();
        int h0 = fp0.getHeight();


        //int elementCount = w*h;
        //int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        //int globalWorkSize = roundUp(localWorkSize, elementCount);


        // ---------------------------------- //
        // ---- Calculate Redundancy Map ---- //
        // ---------------------------------- //

        IJ.log("Calculating redundancy...please wait...");

        int rounds = 5; // How many scale levels should be analyzed
        if(multiScale == 0){

            int scaleFactor = 1;

            if(useTime == 0){

                // --------------------------------------------------------------------------- //
                // ---- Stabilize noise variance using the Generalized Anscombe transform ---- //
                // --------------------------------------------------------------------------- //
                GATMinimizer2D minimizer = new GATMinimizer2D(refPixels0, w0, h0, 0, 100, 0);
                minimizer.run();

                refPixels0 = VarianceStabilisingTransform2D_.getGAT(refPixels0, minimizer.gain, minimizer.sigma, minimizer.offset);


                // ------------------- //
                // ---- Normalize ---- //
                // ------------------- //

                // Cast to "double" type
                double[] refPixelsDouble = new double[w0*h0];
                for(int i=0; i<w0*h0; i++){
                    refPixelsDouble[i] = (double)refPixels0[i];
                }

                // Find min and max
                double imgMin = Double.MAX_VALUE;
                double imgMax = -Double.MAX_VALUE;
                for(int i=0; i<w0*h0; i++){
                    double pixelValue = refPixelsDouble[i];
                    if(pixelValue<imgMin){
                        imgMin = pixelValue;
                    }
                    if(pixelValue>imgMax){
                        imgMax = pixelValue;
                    }
                }

                // Remap pixels
                for(int i=0; i<w0*h0; i++) {
                    refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + (double)EPSILON);
                }

                // Cast back to float
                for(int i=0; i<w0*h0; i++){
                    refPixels0[i] = (float)refPixelsDouble[i];
                }


                // ---------------------- //
                // ---- Run analysis ---- //
                // ---------------------- //

                GlobalRedundancy red0 = new GlobalRedundancy(refPixels0, w0, h0, bW, bH, bRW, bRH, patchSize, EPSILON, context, queue, scaleFactor, filterConstant, 1, w0, h0, metric);
                red0.run();
            }else{
                int nFrames = imp0.getNFrames();
                ImageStack finalStack = new ImageStack(w0, h0, nFrames);

                if(nFrames == 1){
                    IJ.error("No time dimension found. Make sure the image is a stack and Z/T dimensions are not swapped");
                    return;
                }


                // -------------------------------- //
                // ---- Convert image to stack ---- //
                // -------------------------------- //

                ImageStack ims = imp0.getStack();

                // ----------------- //
                // ---- Analyze ---- //
                // ----------------- //
                for(int frame=1; frame<=nFrames; frame++){

                    IJ.log("Processing frame "+frame+"/"+nFrames);


                    // --------------------------------------------------------------------- //
                    // ---- Stabilise variance using the Generalized Anscombe transform ---- //
                    // --------------------------------------------------------------------- //

                    refPixels0 = (float[]) ims.getProcessor(frame).convertToFloatProcessor().getPixels();

                    GATMinimizer2D minimizer = new GATMinimizer2D(refPixels0, w0, h0, 0, 100, 0);
                    minimizer.run();

                    refPixels0 = (float[]) ims.getProcessor(frame).convertToFloatProcessor().getPixels();
                    refPixels0 = VarianceStabilisingTransform2D_.getGAT(refPixels0, minimizer.gain, minimizer.sigma, minimizer.offset);


                    // ---------------------- //
                    // ---- Run analysis ---- //
                    // ---------------------- //

                    GlobalRedundancy red0 = new GlobalRedundancy(refPixels0, w0, h0, bW, bH, bRW, bRH, patchSize, EPSILON, context, queue, scaleFactor, filterConstant, 1, w0, h0, metric);
                    red0.run();


                    // ------------------------------------------- //
                    // ---- Get output and add to final stack ---- //
                    // ------------------------------------------- //

                    ImagePlus impTemp = WindowManager.getImage("Redundancy Map - Level 1");
                    FloatProcessor fpTemp = impTemp.getProcessor().convertToFloatProcessor();

                    finalStack.setProcessor(fpTemp, frame);

                    impTemp.close();

                }

                ImagePlus impFinal = new ImagePlus("Redundancy Maps", finalStack);
                impFinal.show();
            }

        }else{
            int scaleFactor = 1;

            // --------------------------------------------------------------------------- //
            // ---- Stabilize noise variance using the Generalized Anscombe transform ---- //
            // --------------------------------------------------------------------------- //

            GATMinimizer2D minimizer = new GATMinimizer2D(refPixels0, w0, h0, 0, 100, 0);
            minimizer.run();

            refPixels0 = VarianceStabilisingTransform2D_.getGAT(refPixels0, minimizer.gain, minimizer.sigma, minimizer.offset);


            // --------------------------- //
            // ---- Pre-process image ---- //
            // --------------------------- //

            // Cast to "double" type
            double[] refPixelsDouble = new double[w0*h0];
            for(int i=0; i<w0*h0; i++){
                refPixelsDouble[i] = (double)refPixels0[i];
            }

            // Get min and max
            double imgMin = Double.MAX_VALUE;
            double imgMax = -Double.MAX_VALUE;
            for(int i=0; i<w0*h0; i++){
                double pixelValue = refPixelsDouble[i];
                if(pixelValue<imgMin){
                    imgMin = pixelValue;
                }
                if(pixelValue>imgMax){
                    imgMax = pixelValue;
                }
            }

            // Normalize
            for(int i=0; i<w0*h0; i++) {
                refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + (double)EPSILON);
            }

            // Cast back to float
            for(int i=0; i<w0*h0; i++){
                refPixels0[i] = (float)refPixelsDouble[i];
            }


            // ------------------------------------- //
            // ---- Calculate global repetition ---- //
            // ------------------------------------- //

            fp0 = new FloatProcessor(w0, h0, refPixels0);

            for(int i=0; i<rounds; i++){

                // Downscale input image
                int w1 = w0 / scaleFactor; // Width of the downscaled image
                int h1 = h0 / scaleFactor; // Height of the downscaled image
                ImagePlus temp = new ImagePlus("temp", fp0); // Clone original image
                FloatProcessor fp1 = temp.getProcessor().convertToFloatProcessor();

                if(scaleFactor>1) {
                    // Sequential blur and downscale until reaching the desired dimensions (skip first iteration)
                    for(int j=0; j<i; j++) {
                        IJ.run(temp, "Gaussian Blur...", "sigma=1"); // Apply gaussian blur (sigma is 1 px for every time we half the dimensions after)
                        fp1 = temp.getProcessor().convertToFloatProcessor(); // Get blurred image processor
                        fp1 = fp1.resize(w1, h1, true).convertToFloatProcessor(); // Downscale blurred image
                        temp = new ImagePlus("temp", fp1);
                    }
                }

                float[] refPixels1 = (float[]) fp1.getPixels(); // Get blurred and downscale pixel array

                // Calculate redundancy map
                GlobalRedundancy red = new GlobalRedundancy(refPixels1, w1, h1, bW, bH, bRW, bRH, patchSize, EPSILON, context, queue, scaleFactor, filterConstant, i+1, w0, h0, metric);
                Thread thread = new Thread(red);
                thread.start();
                try {
                    thread.join(); // Make sure previous thread is finished before starting a new one (avoids overload)
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                scaleFactor *= 2;
            }

        }

        // ------------------------------- //
        // ---- Cleanup GPU resources ---- //
        // ------------------------------- //

        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");


        // -------------------- //
        // ---- Stop timer ---- //
        // -------------------- //

        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " s");
        IJ.log("--------");

    }

    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //

    public static int getOptimalK(double[][] featureVectors, int maxK){
        // Convert feature vectors array into List
        List<DoublePoint> featureVectorList = new ArrayList<>();
        for(double[] featureVector : featureVectors){
            featureVectorList.add(new DoublePoint(featureVector));
        }

        int optimalK = 1;
        double previousWCSS = Double.MAX_VALUE;
        for(int k=1; k<=maxK; k++){
            KMeansPlusPlusClusterer<DoublePoint> kMeans = new KMeansPlusPlusClusterer<>(k, 1000, new EuclideanDistance());
            List<CentroidCluster<DoublePoint>> clusters = kMeans.cluster(featureVectorList);
            double wcss = 0.0;
            for (CentroidCluster<DoublePoint> cluster : clusters) {
                for (DoublePoint featureVector : cluster.getPoints()) {
                    double distance = new EuclideanDistance().compute(cluster.getCenter().getPoint(), featureVector.getPoint());
                    wcss += distance * distance;
                }
            }
            if (wcss > previousWCSS) {
                break;
            }
            previousWCSS = wcss;
            optimalK = k;
        }
        return optimalK;
    }

    public static float[] applyGaussianBlur(float[] input, int width, int height, float sigma) {
        // Create a Gaussian kernel
        int size = (int) Math.ceil(sigma * 3) * 2 + 1;
        float[] kernel = new float[size];
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float x = (float) (i - (size - 1) / 2.0);
            kernel[i] = (float) Math.exp(-x * x / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }

        // Create a temporary array for the blurred image
        float[] temp = new float[input.length];

        // Blur each row
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum1 = 0;
                float sum2 = 0;
                for (int i = 0; i < size; i++) {
                    int x1 = x + i - (size - 1) / 2;
                    if(x1 < 0 || x1 >= width) {
                        continue;
                    }
                    int index = y*width+x1;
                    float value = input[index];
                    float weight = kernel[i];
                    sum1 += value * weight;
                    sum2 += weight;
                }
                int index = y * width + x;
                temp[index] = sum1 / sum2;
            }
        }

        // Blur each column
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                float sum1 = 0;
                float sum2 = 0;
                for (int i = 0; i < size; i++) {
                    int y1 = y + i - (size - 1) / 2;
                    if (y1 < 0 || y1 >= height) {
                        continue;
                    }
                    int index = y1 * width + x;
                    float value = temp[index];
                    float weight = kernel[i];
                    sum1 += value * weight;
                    sum2 += weight;
                }
                int index = y * width + x;
                input[index] = sum1 / sum2;
            }
        }

        return input;
    }
}