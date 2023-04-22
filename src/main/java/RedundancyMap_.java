/**
 * OPENCL INITIALIZATION (CONTEXT, DEVICE, QUEUE) IS BEING HANDLED IN THIS CLASS, WHILE BUFFERS ARE HANDLED IN THE
 * REDUNDANCY CLASS. This is so that we can create the OpenCL context and devices once and calculate redundancy multiple
 * times. Buffers, programs, kernels, etc, resources are managed in the "GlobalRedundancy" class.
 * TODO: Implement progress tracking
 * TODO: check kernels for division by zero
 **/

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.measure.UserFunction;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import jdk.nashorn.internal.objects.Global;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RedundancyMap_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;
    static private CLPlatform clPlatformMaxFlop;
    static private CLCommandQueue queue;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;

        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Redundancy map");
        gd.addNumericField("Box length in pixels: ", 3, 2);
        gd.addCheckbox("Stabilise variance?", false);
        gd.addCheckbox("Rotation invariant?", false);
        gd.addCheckbox("Multi-scale?", false);
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get dialog parameters
        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = bW; // Patch height

        int useGAT = 0; // Use GAT? (0 = no GAT)
        if(gd.getNextBoolean() == true) {
            useGAT = 1;
        }

        int rotInv = 0; // Rotation invariant analysis?
        if(gd.getNextBoolean() == true) {
            rotInv = 1;
        }

        int multiScale = 0; // Multi-scale anlysis?
        if(gd.getNextBoolean() == true){
            multiScale = 1;
        }

        int downScale = 0; // Downscale factor
        int speedUp = 0; // Speed up factor (0 = no speed up)


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
                IJ.log("--------");
                IJ.log("Device name: " + clDevice.getName());
                IJ.log("Device type: " + clDevice.getType());
                IJ.log("Max clock: " + clDevice.getMaxClockFrequency() + " MHz");
                IJ.log("Number of compute units: " + clDevice.getMaxComputeUnits());
                IJ.log("Max work group size: " + clDevice.getMaxWorkGroupSize());
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
            GlobalRedundancy red0 = new GlobalRedundancy(refPixels0, w0, h0, bW, bH, EPSILON, context, queue, speedUp,
                    useGAT, rotInv, 1, w0, h0);
            red0.run();
        }else{
            int scaleFactor = 1;
            for(int i=0; i<rounds; i++){
                // Downscale input image
                int w1 = w0 / scaleFactor; // Width of the downscaled image
                int h1 = h0 / scaleFactor; // Height of the downscaled image
                ImagePlus temp = new ImagePlus("temp", fp0); // Clone original image
                FloatProcessor fp1 = imp0.getProcessor().convertToFloatProcessor(); // Get blurred image processor

                if(scaleFactor>1) {
                    // Sequential blur and downscale until reaching the desired dimensions
                    for(int j=0; j<i; j++) {
                        IJ.run(temp, "Gaussian Blur...", "sigma=1"); // Apply gaussian blur (sigma is 1 px for every time we half the dimensions after)
                        fp1 = temp.getProcessor().convertToFloatProcessor(); // Get blurred image processor
                        fp1 = fp1.resize(w1, h1, true).convertToFloatProcessor(); // Downscale blurred image
                        temp = new ImagePlus("temp", fp1);
                    }
                }

                float[] refPixels1 = (float[]) fp1.getPixels(); // Get blurred and downscale pixel array

                // Calculate redundancy map
                GlobalRedundancy red = new GlobalRedundancy(refPixels1, w1, h1, bW, bH, EPSILON, context, queue, speedUp,
                        useGAT, rotInv, i+1, w0, h0);
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

        // ---- Find sets of coordinates representing each unique redundancy value
        //float[] pearsonUnique = getUniqueValues(red025.pearsonMap, red025.w, red025.h, red025.bRW, red025.bRH);
        //int[] pearsonUniqueCoords = getUniqueValueCoordinates(pearsonUnique, red025.pearsonMap, red025.w, red025.h, red025.bRW, red025.bRH);


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

    public static float[] getUniqueValues(float[] inArr, int w, int h, int offsetX, int offsetY){
        // Sort input array
        Arrays.sort(inArr);

        // Get number of unique values
        int count = 1; // Starts at 1 because first value is not checked
        for(int j=offsetY; j<h-offsetY; j++){
            for(int i=offsetX; i<w-offsetX; i++){
                if(inArr[j*w+i] != inArr[(j*w+i)+1]){
                    count++;
                }
            }
        }

        // Get unique values
        float[] outArr = new float[count];
        int index = 0;
        for(int j=offsetY; j<h-offsetY; j++){
            for(int i=offsetX; i<w-offsetX; i++) {
                if (inArr[j*w+i] != inArr[(j*w+i)+1]) {
                    outArr[index] = inArr[j*w+i];
                    index++;
                }
            }
        }
        outArr[index] = inArr[inArr.length-1]; // Add the last unique value, which is not added in the loop

        return outArr;
    }

    public static int[] getUniqueValueCoordinates(float[] uniqueArr, float[] inArr, int w, int h, int bRW, int bRH){
        int[] outArr = new int[uniqueArr.length];
        int flag = 0;
        for(int u=0; u<uniqueArr.length; u++){
            flag = 0;
            for(int j=bRH; j<h-bRH; j++) {
                if (flag == 0) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if (flag == 0 && inArr[j*w+i] == uniqueArr[u]) {
                            outArr[u] = j*w+i;
                            flag = 1;
                            break;
                        }
                    }
                }
            }
        }

        return outArr;
    }

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
    // ADD FUNCTIONS HERE
}