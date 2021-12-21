/**
 * TODO: Make exception for when the plugin is started without an active image
 * TODO: Implement progress tracking
 * TODO: Solve redundancy map having an extra column
 * TODO: Think about
 **/

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static nanoj.core2.NanoJCL.replaceFirst;

public class RedundancyMap_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;
    static private CLProgram programGetStats, programGetWeightMap, programGetMeanPearson, programGetLocalMeans, programGetLocalDeviations;
    static private CLKernel kernelGetStats, kernelGetWeightMap, kernelGetMeanPearson, kernelGetLocalMeans, kernelGetLocalDeviations;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalDeviations, clWeightMap, clPearsonMap;

    @Override
    public void run(String s) {

        // ---- Get reference image and some parameters ----
        ImagePlus imp0 = WindowManager.getCurrentImage();
        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = imp0.getWidth();
        int h = imp0.getHeight();
        float sigma = 1.7F; // TODO: This should be the noise STDDEV, which can be taken from a dark patch in the image
        float filterParamSq = (float) pow(0.4 * sigma, 2);

        // ---- Patch parameters ----
        int bW = 3; // Width
        int bH = 3; // Height

        // ---- Check devices ----
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

        // ---- Create context ----
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

        // ---- Create buffers ----
        clRefPixels = context.createFloatBuffer(w * h, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(w * h, READ_WRITE);
        clLocalDeviations = context.createFloatBuffer(w * h, READ_WRITE);
        clWeightMap = context.createFloatBuffer(w * h, READ_WRITE);
        clPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);

    /*
        float[] localSums = new float[w * h];
        clLocalSums = context.createFloatBuffer(w * h, READ_WRITE);

        float[] localSqSums = new float[w * h];
        clLocalSqSums = context.createFloatBuffer(w * h, READ_WRITE);



        float[] localVariances = new float[w * h];
        clLocalVariances = context.createFloatBuffer(w * h, READ_WRITE);

        float[] localDeviations = new float[w * h];
        clLocalDeviations = context.createFloatBuffer(w * h, READ_WRITE);




        float[] meanPearsonMap = new float[w * h];
        clMeanPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);
    */

        // ---- Create programs ----
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BW$", "" + bW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BH$", "" + bH);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        String programStringGetLocalDeviations = getResourceAsString(RedundancyMap_.class, "kernelGetLocalDeviations.cl");
        programStringGetLocalDeviations = replaceFirst(programStringGetLocalDeviations, "$WIDTH$", "" + w);
        programStringGetLocalDeviations = replaceFirst(programStringGetLocalDeviations, "$HEIGHT$", "" + h);
        programStringGetLocalDeviations = replaceFirst(programStringGetLocalDeviations, "$BW$", "" + bW);
        programStringGetLocalDeviations = replaceFirst(programStringGetLocalDeviations, "$BH$", "" + bH);
        programGetLocalDeviations = context.createProgram(programStringGetLocalDeviations).build();

        int patchSize = bW * bH;
        String programStringGetWeightMap = getResourceAsString(RedundancyMap_.class, "kernelGetWeightMap.cl");
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$WIDTH$", "" + w);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$HEIGHT$", "" + h);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$BW$", "" + bW);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$BH$", "" + bH);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$SIGMA$", "" + sigma);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$PATCH_SIZE$", "" + patchSize);
        programGetWeightMap = context.createProgram(programStringGetWeightMap).build();


    /*
        // Create getStatsProgram
        String programStringGetStats = getResourceAsString(RedundancyMap_.class, "kernelGetStats.cl");
        programStringGetStats = replaceFirst(programStringGetStats, "$REFPIXELS$", "" + refPixels);
        programStringGetStats = replaceFirst(programStringGetStats, "$WIDTH$", "" + w);
        programStringGetStats = replaceFirst(programStringGetStats, "$HEIGHT$", "" + h);
        programStringGetStats = replaceFirst(programStringGetStats, "$BW$", "" + bW);
        programStringGetStats = replaceFirst(programStringGetStats, "$BH$", "" + bH);
        programGetStats = context.createProgram(programStringGetStats).build();



        // Create getMeanPearson program
        String programStringGetMeanPearson = getResourceAsString(RedundancyMap_.class, "kernelGetMeanPearson.cl");
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$WIDTH$", "" + w);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$HEIGHT$", "" + h);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$BW$", "" + bW);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$BH$", "" + bH);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$SIGMA$", "" + sigma);
        programGetMeanPearson = context.createProgram(programStringGetMeanPearson).build();
    */

        // ---- Fill buffers ----
        fillBufferWithFloatArray(clRefPixels, refPixels);
        //fillBufferWithFloatArray(clLocalSums, localSums);
        //fillBufferWithFloatArray(clLocalSqSums, localSqSums);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localDeviations = new float[w * h];
        fillBufferWithFloatArray(clLocalDeviations, localDeviations);

        float[] weightMap = new float[w * h];
        fillBufferWithFloatArray(clWeightMap, weightMap);

        float[] pearsonMap = new float[w * h];
        fillBufferWithFloatArray(clPearsonMap, pearsonMap);

        // ---- Create kernels ----
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");
        kernelGetLocalDeviations = programGetLocalDeviations.createCLKernel("kernelGetLocalDeviations");
        kernelGetWeightMap = programGetWeightMap.createCLKernel("kernelGetWeightMap");

        // ---- Set kernel arguments ----
        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);

        argn = 0;
        kernelGetLocalDeviations.setArg(argn++, clRefPixels);
        kernelGetLocalDeviations.setArg(argn++, clLocalDeviations);

        argn = 0;
        kernelGetWeightMap.setArg(argn++, clRefPixels);
        kernelGetWeightMap.setArg(argn++, clLocalMeans);
        kernelGetWeightMap.setArg(argn++, clWeightMap);
        kernelGetWeightMap.setArg(argn++, clPearsonMap);
        kernelGetWeightMap.setNullArg(argn++, 4);
        kernelGetWeightMap.setNullArg(argn++, 4);
        kernelGetWeightMap.setNullArg(argn++, 4);
        kernelGetWeightMap.setNullArg(argn++, 4);

        // ---- Create command queue ----
        queue = chosenDevice.createCommandQueue();

        int elementCount = w * h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        // ---- Calculate local means map ----
        IJ.log("Calculating local means...");
        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalMeans, false);
        queue.put1DRangeKernel(kernelGetLocalMeans, 0, globalWorkSize, localWorkSize);
        queue.finish();
        queue.putReadBuffer(clLocalMeans, true);
        for (int a = 0; a < localMeans.length; a++) {
            localMeans[a] = clLocalMeans.getBuffer().get(a);
        }
        queue.finish();

        kernelGetLocalMeans.release();
        programGetLocalMeans.release();

        IJ.log("Done!");
        IJ.log("--------");

        // ---- Calculate local standard deviations map ----
        IJ.log("Calculating local standard deviations...");
        queue.putWriteBuffer(clLocalDeviations, false);
        queue.put1DRangeKernel(kernelGetLocalDeviations, 0, globalWorkSize, localWorkSize);
        queue.finish();
        queue.putReadBuffer(clLocalDeviations, true);
        for (int b = 0; b < localDeviations.length; b++) {
            localDeviations[b] = clLocalDeviations.getBuffer().get(b);
        }
        queue.finish();

        clLocalDeviations.release();
        kernelGetLocalDeviations.release();
        programGetLocalDeviations.release();

        IJ.log("Done!");
        IJ.log("--------");

        // Calculate weight map
        long start = System.currentTimeMillis();

        IJ.log("Calculating weight map...");
        queue.putWriteBuffer(clWeightMap, false);
        queue.put1DRangeKernel(kernelGetWeightMap, 0, globalWorkSize, localWorkSize);
        queue.finish();
        queue.putReadBuffer(clPearsonMap, true);
        for (int c = 0; c < pearsonMap.length; c++) {
            pearsonMap[c] = clPearsonMap.getBuffer().get(c);
        }
        queue.finish();
        
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Time taken to calculate: " + elapsedTime + " ms");
        IJ.log("--------");
//        int nXBlocks = w/128 + ((w%128==0)?0:1);
//        int nYBlocks = h/128 + ((h%128==0)?0:1);
//        for (int nYB=0; nYB<nYBlocks; nYB++) {
//            int yWorkSize = min(128, h-nYB*128);
//            for (int nXB=0; nXB<nXBlocks; nXB++) {
//                int xWorkSize = min(128, w-nXB*128);
//                queue.put2DRangeKernel(kernelGetWeightMap, nXB*128, nYB*128, xWorkSize, yWorkSize, 0, 0);
//                queue.finish();
//            }
//        }
//        queue.putReadBuffer(clWeightMap, true);
//        queue.finish();
//        for (int c = 0; c < weightMap.length; c++) {
//            weightMap[c] = clWeightMap.getBuffer().get(c);
//        }


    /*
        System.out.println("Calculating local statistics...");
        IJ.log("Calculating local statistics...");
        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalSums, false);
        queue.putWriteBuffer(clLocalSqSums, false);
        queue.putWriteBuffer(clLocalMeans, false);
        queue.putWriteBuffer(clLocalVariances, false);
        queue.putWriteBuffer(clLocalDeviations, false);
        queue.put1DRangeKernel(kernelGetStats, 0, w * h, 0);
        queue.finish();
        IJ.log("Done!");
        IJ.log("--------");

        // TIMER
        long start = System.currentTimeMillis();
        // ---- Calculate weights and weighted mean pearsons ----
        queue.putWriteBuffer(clWeightMap, false);
        IJ.log("Calculating weight map...");
        // Calculate in chunks to avoid crashing
        int nXBlocks = w/128 + ((w%128==0)?0:1);
        int nYBlocks = h/128 + ((h%128==0)?0:1);
        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(128, h-nYB*128);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(128, w-nXB*128);
                queue.put2DRangeKernel(kernelGetWeightMap, nXB*128, nYB*128, xWorkSize, yWorkSize, 0, 0);
            }
        }

        //queue.finish();
        IJ.log("Done!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Time taken to calculate: " + elapsedTime + " ms");
        IJ.log("--------");
    */

        // Cleanup all resources associated with this context
        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // Display results
        IJ.log("Preparing results for display...");
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMap);
        ImagePlus imp1 = new ImagePlus("Redundancy Map", fp1);
        imp1.show();
        IJ.log("Done!");


    }


    public static void fillBufferWithFloat(CLBuffer<FloatBuffer> clBuffer, float pixel) {
        FloatBuffer buffer = clBuffer.getBuffer();
        buffer.put(pixel);
    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n< pixels.length; n++) {
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
}