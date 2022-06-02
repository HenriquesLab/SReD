/**
 * TODO: Make exception for when the plugin is started without an active image
 * TODO: Implement progress tracking
 * TODO: check kernels for division by zero
 **/

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.plugin.PlugIn;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageConverter;
import ij.process.ShortProcessor;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static nanoj.core2.NanoJCL.replaceFirst;

public class RedundancyMap_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetNrmseMap, programGetSsimMap,
            programGetHuMap, programGetEntropyMap, programGetPhaseCorrelationMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetNrmseMap, kernelGetSsimMap,
            kernelGetHuMap, kernelGetEntropyMap, kernelGetPhaseCorrelationMap;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clNrmseMap, clMaeMap,
            clSsimMap, clHuMap, clEntropyMap, clPhaseCorrelationMap;
    private CLBuffer<ShortBuffer> clRefPixels8Bit;


    @Override
    public void run(String s) {
        // ---- Start timer ----
        long start = System.currentTimeMillis();

        // ---- Get reference image and some parameters ----
        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        float[] refPixelsRaw = (float[]) fp0.getPixels();
        int w = fp0.getWidth();
        int h = fp0.getHeight();


        float sigma = 1.7f; // TODO: This should be the noise STDDEV, which can be taken from a dark patch in the image
        float filterParamSq = (float) pow(0.4 * sigma, 2);

        // ---- Patch parameters ----
        int bW = 3; // Patch width
        int bH = 3; // Patch height
        int patchSize = bW * bH; // Patch area
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int widthWithoutBorders = w - bRW*2;
        int heightWithoutBorders = h - bRH*2;
        int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)

        // ---- Normalize image ----
        float minMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 1, 2);

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

        // ---- Create buffers ----
        clRefPixels = context.createFloatBuffer(w * h, READ_ONLY);
        clRefPixels8Bit = context.createShortBuffer(w * h, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(w * h, READ_WRITE);
        clLocalStds = context.createFloatBuffer(w * h, READ_WRITE);
        clPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);
        clNrmseMap = context.createFloatBuffer(w * h, READ_WRITE);
        clMaeMap = context.createFloatBuffer(w * h, READ_WRITE);
        clSsimMap = context.createFloatBuffer(w * h, READ_WRITE);
        clHuMap = context.createFloatBuffer(w * h, READ_WRITE);
        clEntropyMap = context.createFloatBuffer(w * h, READ_WRITE);
        clPhaseCorrelationMap = context.createFloatBuffer(w * h, READ_WRITE);

        // ---- Create programs ----
        // Local means map
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BW$", "" + bW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BH$", "" + bH);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRW$", "" + bRW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRH$", "" + bRH);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        // Weighted mean Pearson correlation coefficient map
        String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetPearsonMap.cl");
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BW$", "" + bW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BH$", "" + bH);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRW$", "" + bRW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRH$", "" + bRH);
        programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

        // Weighted mean NRMSE map
        String programStringGetNrmseMap = getResourceAsString(RedundancyMap_.class, "kernelGetNrmseMap.cl");
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$WIDTH$", "" + w);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$HEIGHT$", "" + h);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BW$", "" + bW);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BH$", "" + bH);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BRW$", "" + bRW);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BRH$", "" + bRH);
        programGetNrmseMap = context.createProgram(programStringGetNrmseMap).build();

        // Weighted mean SSIM map
        String programStringGetSsimMap = getResourceAsString(RedundancyMap_.class, "kernelGetSsimMap.cl");
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$WIDTH$", "" + w);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$HEIGHT$", "" + h);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BW$", "" + bW);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BH$", "" + bH);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BRW$", "" + bRW);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BRH$", "" + bRH);
        programGetSsimMap = context.createProgram(programStringGetSsimMap).build();

        // Hu map
        String programStringGetHuMap = getResourceAsString(RedundancyMap_.class, "kernelGetHuMap.cl");
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$WIDTH$", "" + w);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$HEIGHT$", "" + h);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BW$", "" + bW);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BH$", "" + bH);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BRW$", "" + bRW);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BRH$", "" + bRH);
        programGetHuMap = context.createProgram(programStringGetHuMap).build();

        // Entropy map
        String programStringGetEntropyMap = getResourceAsString(RedundancyMap_.class, "kernelGetEntropyMap.cl");
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$WIDTH$", "" + w);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$HEIGHT$", "" + h);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BW$", "" + bW);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BH$", "" + bH);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BRW$", "" + bRW);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BRH$", "" + bRH);
        programGetEntropyMap = context.createProgram(programStringGetEntropyMap).build();

        // Phase Correlation Map
        String programStringGetPhaseCorrelationMap = getResourceAsString(RedundancyMap_.class, "kernelGetPhaseCorrelationMap.cl");
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$WIDTH$", "" + w);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$HEIGHT$", "" + h);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BW$", "" + bW);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BH$", "" + bH);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BRW$", "" + bRW);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BRH$", "" + bRH);
        programGetPhaseCorrelationMap = context.createProgram(programStringGetPhaseCorrelationMap).build();


        // ---- Fill buffers ----
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[w*h];
        fillBufferWithFloatArray(clLocalStds, localStds);

        float[] pearsonMap = new float[w * h];
        fillBufferWithFloatArray(clPearsonMap, pearsonMap);

        float[] nrmseMap = new float[w * h];
        fillBufferWithFloatArray(clNrmseMap, nrmseMap);

        float[] maeMap = new float[w * h];
        fillBufferWithFloatArray(clMaeMap, maeMap);

        float[] ssimMap = new float[w * h];
        fillBufferWithFloatArray(clSsimMap, ssimMap);

        float[] huMap = new float[w * h];
        fillBufferWithFloatArray(clHuMap, huMap);

        float[] entropyMap = new float[w * h];
        fillBufferWithFloatArray(clEntropyMap, entropyMap);

        float[] phaseCorrelationMap = new float[w * h];
        fillBufferWithFloatArray(clPhaseCorrelationMap, phaseCorrelationMap);

        // Create 8-bit duplicate of reference image (because entropy calculation is better with 8-bit data)
        ImagePlus imp8Bit = imp0.duplicate();
        imp8Bit.setTitle("Variance-stabilized image (8-bit)");
        ImageConverter ic = new ImageConverter(imp8Bit);
        ic.convertToGray8();
        imp8Bit.updateAndDraw();
        ic.convertToGray16();
        imp8Bit.updateAndDraw();
        //FloatProcessor fp8Bit =imp8Bit.getProcessor().convertToFloatProcessor();
        //float[] refPixels8Bit = (float[]) fp8Bit.getPixels();
        //fillBufferWithFloatArray(clRefPixels8Bit, refPixels8Bit);
        ShortProcessor bp = imp8Bit.getProcessor().convertToShortProcessor();
        short[] refPixels8Bit = (short[]) bp.getPixels();
        fillBufferWithShortArray(clRefPixels8Bit, refPixels8Bit);

        // ---- Create kernels ----
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");
        kernelGetPearsonMap = programGetPearsonMap.createCLKernel("kernelGetPearsonMap");
        kernelGetNrmseMap = programGetNrmseMap.createCLKernel("kernelGetNrmseMap");
        kernelGetSsimMap = programGetSsimMap.createCLKernel("kernelGetSsimMap");
        kernelGetHuMap = programGetHuMap.createCLKernel("kernelGetHuMap");
        kernelGetEntropyMap = programGetEntropyMap.createCLKernel("kernelGetEntropyMap");
        kernelGetPhaseCorrelationMap = programGetPhaseCorrelationMap.createCLKernel("kernelGetPhaseCorrelationMap");

        // ---- Set kernel arguments ----
        // Local means map
        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);
        kernelGetLocalMeans.setArg(argn++, clLocalStds);

        // Weighted mean Pearson correlation coefficient map
        argn = 0;
        kernelGetPearsonMap.setArg(argn++, clRefPixels);
        kernelGetPearsonMap.setArg(argn++, clLocalMeans);
        kernelGetPearsonMap.setArg(argn++, clLocalStds);
        kernelGetPearsonMap.setArg(argn++, clPearsonMap);

        // Weighted mean NRMSE map
        argn = 0;
        kernelGetNrmseMap.setArg(argn++, clRefPixels);
        kernelGetNrmseMap.setArg(argn++, clLocalMeans);
        kernelGetNrmseMap.setArg(argn++, clLocalStds);
        kernelGetNrmseMap.setArg(argn++, clNrmseMap);
        kernelGetNrmseMap.setArg(argn++, clMaeMap);

        // Weighted mean SSIM map
        argn = 0;
        kernelGetSsimMap.setArg(argn++, clRefPixels);
        kernelGetSsimMap.setArg(argn++, clLocalMeans);
        kernelGetSsimMap.setArg(argn++, clLocalStds);
        kernelGetSsimMap.setArg(argn++, clSsimMap);

        // Hu map
        argn = 0;
        kernelGetHuMap.setArg(argn++, clRefPixels);
        kernelGetHuMap.setArg(argn++, clLocalMeans);
        kernelGetHuMap.setArg(argn++, clLocalStds);
        kernelGetHuMap.setArg(argn++, clHuMap);

        // Entropy map
        argn = 0;
        kernelGetEntropyMap.setArg(argn++, clRefPixels8Bit);
        kernelGetEntropyMap.setArg(argn++, clLocalMeans);
        kernelGetEntropyMap.setArg(argn++, clLocalStds);
        kernelGetEntropyMap.setArg(argn++, clEntropyMap);

        // Phase correlation map
        argn = 0;
        kernelGetPhaseCorrelationMap.setArg(argn++, clRefPixels);
        kernelGetPhaseCorrelationMap.setArg(argn++, clLocalMeans);
        kernelGetPhaseCorrelationMap.setArg(argn++, clLocalStds);
        kernelGetPhaseCorrelationMap.setArg(argn++, clPhaseCorrelationMap);

        // ---- Create command queue ----
        queue = chosenDevice.createCommandQueue();

        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        // ---- Calculate local means map ----
        IJ.log("Calculating redundancy...please wait...");

        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalMeans, false);

        int nXBlocks = w/64 + ((w%64==0)?0:1);
        int nYBlocks = h/64 + ((h%64==0)?0:1);
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating local means... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetLocalMeans, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the local means map back from the GPU ----
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                localMeans[y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
            }
        }
        queue.finish();

        // ---- Read the local stds map back from the GPU ----
        queue.putReadBuffer(clLocalStds, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                localStds[y*w+x] = clLocalStds.getBuffer().get(y*w+x);
            }
        }
        queue.finish();

        kernelGetLocalMeans.release();
        programGetLocalMeans.release();

        // ---- Calculate weighted mean Pearson's map ----
        queue.putWriteBuffer(clPearsonMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Pearson's correlations... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetPearsonMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the Pearson's map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clPearsonMap, true);
        for (int y = 0; y<h; y++) {
            for(int x=0; x<w; x++) {
                pearsonMap[y*w+x] = clPearsonMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetPearsonMap.release();
        programGetPearsonMap.release();
        clPearsonMap.release();

        // ---- Calculate weighted mean NRMSE and MAE maps ----
        queue.putWriteBuffer(clNrmseMap, false);
        queue.putWriteBuffer(clMaeMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating NRMSE and MAE... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetNrmseMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }

        queue.finish();

        // ---- Read the NRMSE and MAE maps back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clNrmseMap, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                nrmseMap[y*w+x] = clNrmseMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        queue.putReadBuffer(clMaeMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                maeMap[y*w+x] = clMaeMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetNrmseMap.release();
        programGetNrmseMap.release();
        clNrmseMap.release();
        clMaeMap.release();

        // ---- Calculate weighted mean SSIM map ----
        queue.putWriteBuffer(clSsimMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating SSIM... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetSsimMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the SSIM map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clSsimMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                ssimMap[y*w+x] = clSsimMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetSsimMap.release();
        programGetSsimMap.release();
        clSsimMap.release();

        // ---- Calculate Hu map ----
        queue.putWriteBuffer(clHuMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Hu... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetHuMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the Hu map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clHuMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                huMap[y*w+x] = clHuMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetHuMap.release();
        programGetHuMap.release();
        clHuMap.release();

        /*
        // ---- Calculate entropy map ----
        queue.putWriteBuffer(clEntropyMap, false);
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating entropy... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetEntropyMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the entropy map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clEntropyMap, true);
        for (int l=0; l<h; l++) {
            for (int m=0; m<w; m++) {
                entropyMap[l*w+m] = clEntropyMap.getBuffer().get(l*w+m) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetEntropyMap.release();
        programGetEntropyMap.release();
        clEntropyMap.release();

        IJ.log("Done!");
        IJ.log("--------");
*/
        // ---- Calculate Phase correlation map ----
        queue.putWriteBuffer(clPhaseCorrelationMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Phase correlations... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetPhaseCorrelationMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
            }
        }

        queue.finish();

        // ---- Read the Phase correlations map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clPhaseCorrelationMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                phaseCorrelationMap[y*w+x] = clPhaseCorrelationMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        kernelGetPhaseCorrelationMap.release();
        programGetPhaseCorrelationMap.release();
        clPhaseCorrelationMap.release();

        // ---- Cleanup all resources associated with this context ----
        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // ---- Display results ----
        IJ.log("Preparing results for display...");

        // Pearson's map (normalized to [0,1])
        float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
        float[] pearsonMapNorm = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMap);
        ImagePlus imp1 = new ImagePlus("Pearson's Map", fp1);
        imp1.show();

        // NRMSE map (normalized to [0,1])
        float[] nrmseMinMax = findMinMax(nrmseMap, w, h, bRW, bRH);
        float[] nrmseMapNorm = normalize(nrmseMap, w, h, bRW, bRH, nrmseMinMax, 0, 0);
        FloatProcessor fp2 = new FloatProcessor(w, h, nrmseMapNorm);
        ImagePlus imp2 = new ImagePlus("NRMSE Map", fp2);
        imp2.show();

        // MAE map (normalized to [0,1])
        float[] maeMinMax = findMinMax(maeMap, w, h, bRW, bRH);
        float[] maeMapNorm = normalize(maeMap, w, h, bRW, bRH, maeMinMax, 0, 0);
        FloatProcessor fp3 = new FloatProcessor(w, h, maeMapNorm);
        ImagePlus imp3 = new ImagePlus("MAE Map", fp3);
        imp3.show();

        // SSIM map (normalized to [0,1])
        float[] ssimMinMax = findMinMax(ssimMap, w, h, bRW, bRH);
        float[] ssimMapNorm = normalize(ssimMap, w, h, bRW, bRH, ssimMinMax, 0, 0);
        FloatProcessor fp4 = new FloatProcessor(w, h, ssimMapNorm);
        ImagePlus imp4 = new ImagePlus("SSIM Map", fp4);
        imp4.show();

        // Hu map (normalized to [0,1])
        float[] huMinMax = findMinMax(huMap, w, h, bRW, bRH);
        float[] huMapNorm = normalize(huMap, w, h, bRW, bRH, huMinMax, 0, 0);
        FloatProcessor fp5 = new FloatProcessor(w, h, huMapNorm);
        ImagePlus imp5 = new ImagePlus("Hu Map", fp5);
        imp5.show();
/*
        // Entropy map (normalized to [0,1])
        FloatProcessor fp6 = new FloatProcessor(w, h, entropyMap);
        ImagePlus imp6 = new ImagePlus("Entropy Map", fp6);
        imp6.show();
        IJ.log("Finished!");
*/
        // Phase map (normalized to [0,1])
        float[] phaseMinMax = findMinMax(phaseCorrelationMap, w, h, bRW, bRH);
        float[] phaseMapNorm = normalize(phaseCorrelationMap, w, h, bRW, bRH, phaseMinMax, 0, 0);
        FloatProcessor fp6 = new FloatProcessor(w, h, phaseMapNorm);
        ImagePlus imp6 = new ImagePlus("Phase Map", fp6);
        imp6.show();

        // ---- Stop timer ----
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }

    // ---- USER FUNCTIONS ----

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

    public static void fillBufferWithShortArray(CLBuffer<ShortBuffer> clBuffer, short[] pixels) {
        ShortBuffer buffer = clBuffer.getBuffer();
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

    public static float[] findMinMax(float[] inputArray, int w, int h, int offsetX, int offsetY){
        float[] minMax = {inputArray[offsetY*w+offsetX], inputArray[offsetY*w+offsetX]};

        for(int j=offsetY+1; j<h-offsetY; j++){
            for(int i=offsetX+1; i<w-offsetX; i++){
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