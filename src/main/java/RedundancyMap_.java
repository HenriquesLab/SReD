/**
 * TODO: Make exception for when the plugin is started without an active image
 * TODO: Implement progress tracking
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
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetRmseMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetRmseMap;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clPearsonMap, clRmseMap, clMaeMap;

    @Override
    public void run(String s) {

        // ---- Start timer ----
        long start = System.currentTimeMillis();

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
        int patchSize = bW * bH;
        int sizeWithoutBorders = (w-2)*(h-2); // TODO: MAke this dynamic
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
        clPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);
        clRmseMap = context.createFloatBuffer(w * h, READ_WRITE);
        clMaeMap = context.createFloatBuffer(w * h, READ_WRITE);

        // ---- Create programs ----
        // Local means map
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BW$", "" + bW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BH$", "" + bH);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        // Weighted mean Pearson correlation coefficient map
        String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetPearsonMap.cl");
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BW$", "" + bW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BH$", "" + bH);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
        programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

        // Weighted mean RMSE map
        String programStringGetRmseMap = getResourceAsString(RedundancyMap_.class, "kernelGetRmseMap.cl");
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$WIDTH$", "" + w);
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$HEIGHT$", "" + h);
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$BW$", "" + bW);
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$BH$", "" + bH);
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetRmseMap = replaceFirst(programStringGetRmseMap, "$PATCH_SIZE$", "" + patchSize);
        programGetRmseMap = context.createProgram(programStringGetRmseMap).build();

        // ---- Fill buffers ----
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] pearsonMap = new float[w * h];
        fillBufferWithFloatArray(clPearsonMap, pearsonMap);

        float[] rmseMap = new float[w * h];
        fillBufferWithFloatArray(clRmseMap, rmseMap);

        float[] maeMap = new float[w * h];
        fillBufferWithFloatArray(clMaeMap, maeMap);

        // ---- Create kernels ----
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");
        kernelGetPearsonMap = programGetPearsonMap.createCLKernel("kernelGetPearsonMap");
        kernelGetRmseMap = programGetRmseMap.createCLKernel("kernelGetRmseMap");

        // ---- Set kernel arguments ----
        // Local means map
        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);

        // Weighted mean Pearson correlation coefficient map
        argn = 0;
        kernelGetPearsonMap.setArg(argn++, clRefPixels);
        kernelGetPearsonMap.setArg(argn++, clLocalMeans);
        kernelGetPearsonMap.setArg(argn++, clPearsonMap);

        // Weighted mean RMSE map
        argn = 0;
        kernelGetRmseMap.setArg(argn++, clRefPixels);
        kernelGetRmseMap.setArg(argn++, clLocalMeans);
        kernelGetRmseMap.setArg(argn++, clRmseMap);
        kernelGetRmseMap.setArg(argn++, clMaeMap);

        // ---- Create command queue ----
        queue = chosenDevice.createCommandQueue();

        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        // ---- Calculate local means map ----
        IJ.log("Calculating redundancy...");
        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalMeans, false);
        queue.put1DRangeKernel(kernelGetLocalMeans, 0, w*h, 0);
        queue.finish();
        queue.putReadBuffer(clLocalMeans, true);
        for (int a = 0; a < localMeans.length; a++) {
            localMeans[a] = clLocalMeans.getBuffer().get(a);
        }
        queue.finish();

        kernelGetLocalMeans.release();
        programGetLocalMeans.release();

        //IJ.log("Done!");
        //IJ.log("--------");

        // ---- Calculate weighted mean Pearson's map ----
        queue.putWriteBuffer(clPearsonMap, false);
/*
        int nXBlocks = w/128 + ((w%128==0)?0:1);
        int nYBlocks = h/128 + ((h%128==0)?0:1);
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(128, h-nYB*128);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(128, w-nXB*128);
                showStatus("Calculating redundancy... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetWeightMap, nXB*128, nYB*128, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }
        */
        queue.put2DRangeKernel(kernelGetPearsonMap, 0, 0, w, h, 0,0);
        queue.finish();

        // ---- Read the Pearson's map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clPearsonMap, true);
        for (int c = 0; c < pearsonMap.length; c++) {
            pearsonMap[c] = clPearsonMap.getBuffer().get(c) / sizeWithoutBorders;
            queue.finish();
        }

        //kernelGetPearsonMap.release();
        //programGetPearsonMap.release();
        //clPearsonMap.release();

        // ---- Calculate weighted mean RMSE map ----
        queue.putWriteBuffer(clRmseMap, false);
        queue.put2DRangeKernel(kernelGetRmseMap, 0, 0, w, h, 0,0);
        queue.finish();

        // ---- Read the RMSE and MAE maps back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clRmseMap, true);
        for (int d = 0; d < rmseMap.length; d++) {
            rmseMap[d] = clRmseMap.getBuffer().get(d) / sizeWithoutBorders;
            queue.finish();
        }

        queue.putReadBuffer(clMaeMap, true);
        for (int e = 0; e < maeMap.length; e++) {
            maeMap[e] = clMaeMap.getBuffer().get(e) / sizeWithoutBorders;
            queue.finish();
        }

        //kernelGetRmseMap.release();
        //programGetRmseMap.release();
        //clRmseMap.release();

        IJ.log("Done!");
        IJ.log("--------");

        // ---- Cleanup all resources associated with this context ----
        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // ---- Display results ----
        IJ.log("Preparing results for display...");

        // Pearson's map
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMap);
        ImagePlus imp1 = new ImagePlus("Pearson's Map", fp1);
        imp1.show();

        // RMSE map
        FloatProcessor fp2 = new FloatProcessor(w, h, rmseMap);
        ImagePlus imp2 = new ImagePlus("RMSE Map", fp2);
        imp2.show();

        // MAE map
        FloatProcessor fp3 = new FloatProcessor(w, h, maeMap);
        ImagePlus imp3 = new ImagePlus("MAE Map", fp3);
        imp3.show();
        IJ.log("Done!");

        // ---- Stop timer ----
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
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