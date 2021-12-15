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
    static private CLProgram programGetStats, programGetWeightMap, programGetMeanPearson;
    static private CLKernel kernelGetStats, kernelGetWeightMap, kernelGetMeanPearson;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalSums, clLocalMeans, clLocalVariances, clLocalDeviations,
            clLocalSqSums, clWeightMap, clMeanPearsonMap;

    @Override
    public void run(String s) {

        // ---- Get reference image and some parameters ----
        ImagePlus imp0 = WindowManager.getCurrentImage();
        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = imp0.getWidth();
        int h = imp0.getHeight();
        float sigma = 1.7F; // TODO: This should be the noise STDDEV, which can be taken from a dark patch in the image
        float filterParamSq = (float) pow(0.4*sigma, 2);

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

        for(CLPlatform allPlatform : allPlatforms) {
            CLDevice[] allCLdeviceOnThisPlatform = allPlatform.listCLDevices();

            for(CLDevice clDevice : allCLdeviceOnThisPlatform) {
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
        for (int i=0; i<allDevices.length; i++) {
            if (allDevices[i].getType() == CLDevice.Type.GPU) {
                hasGPU = true;
            }
        }
        CLDevice chosenDevice;
        if (hasGPU) {
            chosenDevice = context.getMaxFlopsDevice(CLDevice.Type.GPU);
        }else{
            chosenDevice = context.getMaxFlopsDevice();
        }

        IJ.log("Chosen device: " + chosenDevice.getName());
        IJ.log("--------");

        // ---- Create buffers ----
        clRefPixels = context.createFloatBuffer(w * h, READ_ONLY);

        float localSums[] = new float[w * h];
        clLocalSums = context.createFloatBuffer(w * h, READ_WRITE);

        float localSqSums[] = new float[w * h];
        clLocalSqSums = context.createFloatBuffer(w * h, READ_WRITE);

        float localMeans[] = new float[w * h];
        clLocalMeans = context.createFloatBuffer(w * h, READ_WRITE);

        float localVariances[] = new float[w * h];
        clLocalVariances = context.createFloatBuffer(w * h, READ_WRITE);

        float localDeviations[] = new float[w * h];
        clLocalDeviations = context.createFloatBuffer(w * h, READ_WRITE);

        float[] weightMap = new float[w * h];
        clWeightMap = context.createFloatBuffer(w * h, READ_WRITE);

        float[] meanPearsonMap = new float[w * h];
        clMeanPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);



        // ---- Create programs ----
        // Create getStatsProgram
        String programStringGetStats = getResourceAsString(RedundancyMap_.class, "kernelGetStats.cl");
        programStringGetStats = replaceFirst(programStringGetStats, "$REFPIXELS$", "" + refPixels);
        programStringGetStats = replaceFirst(programStringGetStats, "$WIDTH$", "" + w);
        programStringGetStats = replaceFirst(programStringGetStats, "$HEIGHT$", "" + h);
        programStringGetStats = replaceFirst(programStringGetStats, "$BW$", "" + bW);
        programStringGetStats = replaceFirst(programStringGetStats, "$BH$", "" + bH);
        programGetStats = context.createProgram(programStringGetStats).build();

        // Create getWeightMap program
        String programStringGetWeightMap = getResourceAsString(RedundancyMap_.class, "kernelGetWeightMap.cl");
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$WIDTH$", "" + w);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$HEIGHT$", "" + h);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$SIGMA$", "" + sigma);
        programStringGetWeightMap = replaceFirst(programStringGetWeightMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);

        programGetWeightMap = context.createProgram(programStringGetWeightMap).build();


        // Create getMeanPearson program
        String programStringGetMeanPearson = getResourceAsString(RedundancyMap_.class, "kernelGetMeanPearson.cl");
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$WIDTH$", "" + w);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$HEIGHT$", "" + h);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$BW$", "" + bW);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$BH$", "" + bH);
        programStringGetMeanPearson = replaceFirst(programStringGetMeanPearson, "$SIGMA$", "" + sigma);
        programGetMeanPearson = context.createProgram(programStringGetMeanPearson).build();

        // ---- Fill buffers ----
        fillBufferWithFloatArray(clRefPixels, refPixels);
        fillBufferWithFloatArray(clLocalSums, localSums);
        fillBufferWithFloatArray(clLocalSqSums, localSqSums);
        fillBufferWithFloatArray(clLocalMeans, localMeans);
        fillBufferWithFloatArray(clLocalVariances, localVariances);
        fillBufferWithFloatArray(clLocalDeviations, localDeviations);
        fillBufferWithFloatArray(clMeanPearsonMap, meanPearsonMap);
        fillBufferWithFloatArray(clWeightMap, weightMap);

        // ---- Create kernels ----
        kernelGetStats = programGetStats.createCLKernel("kernelGetStats");
        kernelGetWeightMap = programGetWeightMap.createCLKernel("kernelGetWeightMap");
        //kernelGetMeanPearson = programGetMeanPearson.createCLKernel("kernelGetMeanPearson");


        // ---- Set kernel arguments ----
        int argn = 0;
        kernelGetStats.setArg(argn++, clRefPixels);
        kernelGetStats.setArg(argn++, clLocalSums);
        kernelGetStats.setArg(argn++, clLocalSqSums);
        kernelGetStats.setArg(argn++, clLocalMeans);
        kernelGetStats.setArg(argn++, clLocalVariances);
        kernelGetStats.setArg(argn++, clLocalDeviations);

       /* argn = 0;
        kernelGetMeanPearson.setArg(argn++, clRefPixels);
        kernelGetMeanPearson.setArg(argn++, clLocalMeans);
        kernelGetMeanPearson.setArg(argn++, clLocalDeviations);
        kernelGetMeanPearson.setArg(argn++, clWeightMap);
        kernelGetMeanPearson.setArg(argn++, clMeanPearsonMap);
        */

        argn = 0;
        kernelGetWeightMap.setArg(argn++, clLocalMeans);
        kernelGetWeightMap.setArg(argn++, clWeightMap);

        // ---- Create command queue ----
        queue = chosenDevice.createCommandQueue();


        // ---- Calculate local statistics ----
        System.out.println("Calculating local statistics...");
        IJ.log("Calculating local statistics...");
        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalSums, false);
        queue.putWriteBuffer(clLocalSqSums, false);
        queue.putWriteBuffer(clLocalMeans, false);
        queue.putWriteBuffer(clLocalVariances, false);
        queue.putWriteBuffer(clLocalDeviations, false);
       // queue.put1DRangeKernel(kernelGetStats, 0, w*h, 0);
        queue.finish();
        IJ.log("Done!");
        IJ.log("--------");


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
        queue.finish();
        IJ.log("Done!");
        IJ.log("--------");

        // Cleanup all resources associated with this context
        IJ.log("Cleaning up resources...");
        //context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // ---- Display the results ----
        IJ.log("Preparing results for display...");
        queue.putReadBuffer(clMeanPearsonMap, true);
        for(int a=0; a<meanPearsonMap.length; a++) {
            meanPearsonMap[a] = clMeanPearsonMap.getBuffer().get(a);
        }
        FloatProcessor fp1 = new FloatProcessor(w, h, meanPearsonMap);
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