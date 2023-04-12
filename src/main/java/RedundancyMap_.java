/**
 * OPENCL INITIALIZATION (CONTEXT, DEVICE, QUEUE) IS BEING HANDLED IN THIS CLASS, WHILE BUFFERS ARE HANDLED IN THE REDUNDANCY CLASS
 * TODO: Implement progress tracking
 * TODO: check kernels for division by zero
 **/

import com.jogamp.opencl.*;
import distance.Euclidean;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RedundancyMap_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;
    static private CLPlatform clPlatformMaxFlop;
    static private CLCommandQueue queue;

    @Override
    public void run(String s) {

        // ---- Dialog box ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Redundancy map");
        gd.addNumericField("Box length in pixels: ", 3, 2);
        gd.addCheckbox("Stabilise variance?", false);
        gd.addCheckbox("Use Gaussian windows?", false);
        gd.addCheckbox("Use circular patches?", false);
        gd.addNumericField("Sigma: ", 0.5f);
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // ---- Patch parameters ----
        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = bW; // Patch height
        int downScale = 0; // Downscale factor
        int speedUp = 0; // Speed up factor (0 = no speed up)

        int useGAT = 0; // Use GAT (0 = no GAT)
        if(gd.getNextBoolean() == true) {
            useGAT = 1;
        }

        int gaussWind = 0;
        if(gd.getNextBoolean() == true){
            gaussWind = 1;
        }
        float gaussSigma = (float) gd.getNextNumber();

        int circle = 0;
        if(gd.getNextBoolean() == true){
            circle = 1;
        }

        float EPSILON = 0.0000001f;

        // ---- Start timer ----
        long start = System.currentTimeMillis();

        // ---- Initialize OpenCL ----
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

        // ---- Get reference image and some parameters ----
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

        // ---- FIRST ROUND OF REDUNDANCY USING 0.25x IMAGE----
        // Get downscaled image
        if(downScale == 1) {
            // Downscale image
            int w025 = w0 / 2;
            //int w025 = w0;
            int h025 = h0 / 2;
            //int h025 = h0;
            ImagePlus imp025 = imp0.resize(w025, h025, "bicubic");
            FloatProcessor fp025 = imp025.getProcessor().convertToFloatProcessor();
            float[] refPixels025 = (float[]) fp025.getPixels();

            // Calculate redundancy maps
            GlobalRedundancy red025 = new GlobalRedundancy(refPixels025, w025, h025, bW, bH, EPSILON, context, queue,
                    speedUp, useGAT, gaussWind, gaussSigma, circle);
            red025.run();
        }else{
            GlobalRedundancy red0 = new GlobalRedundancy(refPixels0, w0, h0, bW, bH, EPSILON, context, queue, speedUp,
                    useGAT, gaussWind, gaussSigma, circle);
            red0.run();
        }


        // ---- Find sets of coordinates representing each unique redundancy value
        //float[] pearsonUnique = getUniqueValues(red025.pearsonMap, red025.w, red025.h, red025.bRW, red025.bRH);
        //int[] pearsonUniqueCoords = getUniqueValueCoordinates(pearsonUnique, red025.pearsonMap, red025.w, red025.h, red025.bRW, red025.bRH);






        // ---- Cleanup all resources associated with this context ----
        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // ---- Stop timer ----
        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime + " ms");
        IJ.log("--------");

    }

    // ---- USER FUNCTIONS ----
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

    // ADD FUNCTIONS HERE
}