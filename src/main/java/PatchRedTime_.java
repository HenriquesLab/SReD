//TODO: Filling buffer with a patch writes wrong values. Currently the kernels are reading the reference patch from the image buffer based on the patch position. Try to fix this to use a patch written in a buffer.

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.awt.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;


public class PatchRedTime_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetPatchPearson;

    static private CLKernel kernelGetPatchMeans, kernelGetPatchPearson;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPatch, clRefPatchMeanSub, clRefPixels, clLocalMeans, clLocalStds, clPearsonMap;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        long start = System.currentTimeMillis();


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        int nFrames = imp0.getNSlices();
        if (nFrames < 2) {
            IJ.error("A stack was not detected. For single images, please use the \"Block Redundancy\" plugin.");
            return;
        }

        ImageProcessor ipRef = imp0.getProcessor();
        FloatProcessor fpRef = ipRef.convertToFloatProcessor();
        float[] refFrame = (float[]) fpRef.getPixels();

        int w = fpRef.getWidth();
        int h = fpRef.getHeight();
        int wh = w * h;

        // Check if patch is selected
        Roi roi = imp0.getRoi();
        if (roi == null) {
            IJ.error("No ROI selected. Please draw an odd-sized square patch and try again.");
            return;
        }

        // Check if patch dimensions are odd and get patch parameters
        Rectangle rect = ipRef.getRoi(); // Getting ROI from float processor is not working correctly
        int bx = rect.x; // x-coordinate of the top left corner of the rectangle
        int by = rect.y; // y-coordinate of the top left corner of the rectangle
        int bW = rect.width; // Patch width
        int bH = rect.height; // Patch height
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)
        int patchSize = (2*bRW+1) * (2*bRW+1) - (int) ceil((sqrt(2)*bRW)*(sqrt(2)*bRW));
        int centerX = bx + bRW; // Patch center (x-axis)
        int centerY = by + bRH; // Patch center (y-axis)
        System.out.println("Center X: " + centerX + ", Center Y: " + centerY);

        // Verify that selected patch dimensions are odd
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }


        // ----------------------------------------------------------------------------- //
        // ---- Stabilize noise variance using the Generalized Anscombe's transform ---- //
        // ----------------------------------------------------------------------------- //

        // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
        GATMinimizer minimizer = new GATMinimizer(refFrame, w, h, 0, 100, 0);
        minimizer.run();

        // Get gain, sigma and offset from minimizer and transform pixel values
        refFrame = TransformImageByVST_.getGAT(refFrame, minimizer.gain, minimizer.sigma, minimizer.offset);


        // ------------------------------- //
        // ---- Normalize input image ---- //
        // ------------------------------- //

        float minMax[] = findMinMax(refFrame, w, h, 0, 0);
        refFrame = normalize(refFrame, w, h, 0, 0, minMax, 0, 0);


        // ------------------------------------------------- //
        // ---- Get reference patch and some statistics ---- //
        // ------------------------------------------------- //

        // Get reference patch
        float[] refPatch = new float[patchSize];
        int counter = 0;
        float r2 = bRW*bRW;
        for(int y=centerY-bRH; y<=centerY+bRH; y++) {
            for (int x=centerX-bRW; x<=centerX+bRW; x++) {
                //float dx = (float) (x - centerX);
                //float dy = (float) (y - centerY);
                if(x*x+y*y <= r2){
                    refPatch[counter] = refFrame[y*w+x];
                    System.out.println("counter: "+counter);
                    counter++;
                }
            }
        }

        // Get reference patch statistics
        float patchStats[] = meanVarStd(refPatch);
        float mean = patchStats[0];
        //float var = patchStats[1];
        //float std = patchStats[2];

        // Get mean-subtracted  reference patch
        float[] refPatchMeanSub = new float[patchSize];
        for(int i=0; i<patchSize; i++) {
            refPatchMeanSub[i] = refPatch[i] - mean;
        }


        // --------------------------- //
        // ---- Initialise OpenCL ---- //
        // --------------------------- //

        // Check OpenCL devices
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

        // Choose the best device (i.e., Filter out CPUs if GPUs are available (FLOPS calculation was giving higher ratings to CPU vs. GPU))
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
        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create OpenCL buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans = getResourceAsString(PatchRed_.class, "kernelGetPatchMeans.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Create OpenCL kernel and set args
        kernelGetPatchMeans = programGetPatchMeans.createCLKernel("kernelGetPatchMeans");

        int argn = 0;
        kernelGetPatchMeans.setArg(argn++, clRefPixels);
        kernelGetPatchMeans.setArg(argn++, clLocalMeans);
        kernelGetPatchMeans.setArg(argn++, clLocalStds);


        // ------------------------------------------ //
        // ---- Calculate Pearson's correlations ---- //
        // ------------------------------------------ //

        // Create OpenCL buffers
        clRefPatch = context.createFloatBuffer(patchSize, READ_ONLY);
        clRefPatchMeanSub = context.createFloatBuffer(patchSize, READ_ONLY);
        clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchPearson = getResourceAsString(PatchRed_.class, "kernelGetPatchPearson.cl");
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$WIDTH$", "" + w);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$HEIGHT$", "" + h);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_X$", "" + centerX);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_Y$", "" + centerY);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRW$", "" + bRW);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRH$", "" + bRH);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$EPSILON$", "" + EPSILON);
        programGetPatchPearson = context.createProgram(programStringGetPatchPearson).build();

        // Create OpenCL kernel and set args
        kernelGetPatchPearson = programGetPatchPearson.createCLKernel("kernelGetPatchPearson");

        argn = 0;
        kernelGetPatchPearson.setArg(argn++, clRefPixels);
        kernelGetPatchPearson.setArg(argn++, clLocalMeans);
        kernelGetPatchPearson.setArg(argn++, clLocalStds);
        kernelGetPatchPearson.setArg(argn++, clPearsonMap);

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPatch, refPatch);
        fillBufferWithFloatArray(clRefPatchMeanSub, refPatchMeanSub);

        // Calculate
        float[] refPixels;
        float[] localMeans = new float[wh];
        float[] localStds = new float[wh];
        float[] pearsonMap = new float[wh];

        float[][] finalLocalMeans = new float[nFrames][wh];
        float[][] finalLocalStds = new float[nFrames][wh];
        float[][] finalPearsonMap = new float[nFrames][wh];

        ImageStack ims0 = imp0.getStack();
        for(int i=1; i<=nFrames; i++) {
            IJ.log("Calculating redundancy " + i + "/" + nFrames);
            IJ.showStatus("Calculating redundancy " + i + "/" + nFrames);
            // Stabilize noise variance using the Generalized Anscombe's transform
            // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
            ipRef = ims0.getProcessor(i).convertToFloatProcessor();
            fpRef = ipRef.convertToFloatProcessor();

            refPixels = (float[]) fpRef.getPixels();
            minimizer = new GATMinimizer(refPixels, w, h, 0, 100, 0);
            minimizer.run();

            // Get gain, sigma and offset from minimizer and transform pixel values
            refPixels = TransformImageByVST_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);

            // Normalize input image
            minMax = findMinMax(refPixels, w, h, 0, 0);
            refPixels = normalize(refPixels, w, h, 0, 0, minMax, 0, 0);

            // Fill OpenCL buffers
            fillBufferWithFloatArray(clRefPixels, refPixels);
            fillBufferWithFloatArray(clLocalMeans, localMeans);
            fillBufferWithFloatArray(clLocalStds, localStds);

            // Calculate local means and StdDevs
            queue.putWriteBuffer(clRefPixels, false);
            queue.putWriteBuffer(clLocalMeans, false);
            queue.putWriteBuffer(clLocalStds, false);

            queue.put2DRangeKernel(kernelGetPatchMeans, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read the local means map back from the GPU
            queue.putReadBuffer(clLocalMeans, true);
            for (int y=0; y<h; y++) {
                for(int x=0; x<w; x++) {
                    finalLocalMeans[i-1][y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
                }
            }
            queue.finish();

            // Read the local stds map back from the GPU
            queue.putReadBuffer(clLocalStds, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    finalLocalStds[i-1][y*w+x] = clLocalStds.getBuffer().get(y*w+x);
                }
            }
            queue.finish();

            // Calculate Pearson's correlations
            fillBufferWithFloatArray(clPearsonMap, pearsonMap);

            queue.putWriteBuffer(clPearsonMap, false);
            queue.put2DRangeKernel(kernelGetPatchPearson, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's correlations back from the GPU
            queue.putReadBuffer(clPearsonMap, true);
            for (int y = 0; y<h; y++) {
                for(int x=0; x<w; x++) {
                    finalPearsonMap[i-1][y*w+x] = clPearsonMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();
        }

        IJ.log("Done!");
        IJ.log("--------");


        // ------------------------------- //
        // ---- Cleanup GPU resources ---- //
        // ------------------------------- //

        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");


        // ------------------------- //
        // ---- Display results ---- //
        // ------------------------- //

        IJ.log("Preparing results for display...");

        // Pearson map (stack)
        ImageStack imsPearson = new ImageStack(w, h, nFrames);
        for(int i=1; i<=nFrames; i++){
            float[] pearsonMinMax = findMinMax(finalPearsonMap[i-1], w, h, bRW, bRH);
            float[] pearsonMapNorm = normalize(finalPearsonMap[i-1], w, h, bRW, bRH, pearsonMinMax, 0, 0);
            imsPearson.setPixels(pearsonMapNorm, i);
        }

        ImagePlus imp1 = new ImagePlus("Pearson's Map", imsPearson);
        imp1.show();


        // -------------------- //
        // ---- Stop timer ---- //
        // -------------------- //

        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }

    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //

    private float[] meanVarStd (float a[]){
        int n = a.length;
        if (n == 0) return new float[]{0, 0, 0};

        double sum = 0;
        double sq_sum = 0;

        for (int i = 0; i < n; i++) {
            sum += a[i];
            sq_sum += a[i] * a[i];
        }

        double mean = sum / n;
        double variance = abs(sq_sum / n - mean * mean); // abs() solves a bug where negative zeros appeared

        return new float[]{(float) mean, (float) variance, (float) sqrt(variance)};

    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

    // Read a kernel from the resources
    public static String getResourceAsString(Class c, String resourceName) {
        InputStream programStream = c.getResourceAsStream("/" + resourceName);
        String programString = "";

        try {
            programString = inputStreamToString(programStream);
        } catch (IOException var5) {
            var5.printStackTrace();
        }

        return programString;
    }

    public static String inputStreamToString(InputStream inputStream) throws IOException {
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int length;
        while((length = inputStream.read(buffer)) != -1) {
            result.write(buffer, 0, length);
        }
        return result.toString("UTF-8");
    }

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    public static float[] findMinMax(float[] inputArray, int w, int h, int offsetX, int offsetY){
        float[] minMax = {inputArray[offsetY*w+offsetX], inputArray[offsetY*w+offsetX]};

        for(int j=offsetY; j<h-offsetY; j++){
            for(int i=offsetX; i<w-offsetX; i++){
                if(inputArray[j*w+i] < minMax[0]){
                    minMax[0] = inputArray[j*w+i];
                }
                if(inputArray[j*w+i] > minMax[1]){
                    minMax[1] = inputArray[j*w+i];
                }
            }
        }
        return minMax;
    }

    public static float[] normalize(float[] rawPixels, int w, int h, int offsetX, int offsetY, float[] minMax, float tMin, float tMax){
        float rMin = minMax[0];
        float rMax = minMax[1];
        float denominator = rMax - rMin + 0.000001f;
        float factor;

        if(tMax == 0 && tMin == 0){
            factor = 1; // So that the users can say they don't want an artificial range by choosing tMax and tMin = 0
        }else {
            factor = tMax - tMin;
        }
        float[] normalizedPixels = new float[w*h];

        for(int j=offsetY; j<h-offsetY; j++) {
            for (int i=offsetX; i<w-offsetX; i++) {
                normalizedPixels[j*w+i] = (rawPixels[j*w+i]-rMin)/denominator * factor + tMin;
            }
        }
        return normalizedPixels;
    }
}


