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
            programGetHuMap, programGetEntropyMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetNrmseMap, kernelGetSsimMap,
            kernelGetHuMap, kernelGetEntropyMap;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clPearsonMap, clNrmseMap, clMaeMap,
            clSsimMap, clHuMap, clEntropyMap;
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
        int w = imp0.getWidth();
        int h = imp0.getHeight();
        float sigma = 1.7F; // TODO: This should be the noise STDDEV, which can be taken from a dark patch in the image
        float filterParamSq = (float) pow(0.4 * sigma, 2);

        // ---- Patch parameters ----
        int bW = 7; // Patch width
        int bH = 7; // Patch height
        int patchSize = bW * bH; // Patch area
        int offsetX = bW/2; // Offset of the search radius relative to the original image, to avoid borders (x-axis)
        int offsetY = bH/2; // Offset of the search radius relative to the original image, to avoid borders (y-axis)
        int sizeWithoutBorders = (w-offsetX*2)*(h-offsetY*2); // The area of the search field (= image without borders)

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
        clPearsonMap = context.createFloatBuffer(w * h, READ_WRITE);
        clNrmseMap = context.createFloatBuffer(w * h, READ_WRITE);
        clMaeMap = context.createFloatBuffer(w * h, READ_WRITE);
        clSsimMap = context.createFloatBuffer(w * h, READ_WRITE);
        clHuMap = context.createFloatBuffer(w * h, READ_WRITE);
        clEntropyMap = context.createFloatBuffer(w * h, READ_WRITE);

        // ---- Create programs ----
        // Local means map
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BW$", "" + bW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BH$", "" + bH);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$OFFSET_X$", "" + offsetX);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$OFFSET_Y$", "" + offsetY);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        // Weighted mean Pearson correlation coefficient map
        String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetPearsonMap.cl");
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BW$", "" + bW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BH$", "" + bH);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$OFFSET_X$", "" + offsetX);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$OFFSET_Y$", "" + offsetY);
        programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

        // Weighted mean NRMSE map
        String programStringGetNrmseMap = getResourceAsString(RedundancyMap_.class, "kernelGetNrmseMap.cl");
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$WIDTH$", "" + w);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$HEIGHT$", "" + h);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BW$", "" + bW);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BH$", "" + bH);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$OFFSET_X$", "" + offsetX);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$OFFSET_Y$", "" + offsetY);
        programGetNrmseMap = context.createProgram(programStringGetNrmseMap).build();

        // Weighted mean SSIM map
        String programStringGetSsimMap = getResourceAsString(RedundancyMap_.class, "kernelGetSsimMap.cl");
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$WIDTH$", "" + w);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$HEIGHT$", "" + h);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BW$", "" + bW);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BH$", "" + bH);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$OFFSET_X$", "" + offsetX);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$OFFSET_Y$", "" + offsetY);
        programGetSsimMap = context.createProgram(programStringGetSsimMap).build();

        // Hu map
        String programStringGetHuMap = getResourceAsString(RedundancyMap_.class, "kernelGetHuMap.cl");
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$WIDTH$", "" + w);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$HEIGHT$", "" + h);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BW$", "" + bW);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BH$", "" + bH);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$OFFSET_X$", "" + offsetX);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$OFFSET_Y$", "" + offsetY);
        programGetHuMap = context.createProgram(programStringGetHuMap).build();

        // Entropy map
        String programStringGetEntropyMap = getResourceAsString(RedundancyMap_.class, "kernelGetEntropyMap.cl");
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$WIDTH$", "" + w);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$HEIGHT$", "" + h);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BW$", "" + bW);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BH$", "" + bH);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$FILTER_PARAM_SQ$", "" + filterParamSq);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$OFFSET_X$", "" + offsetX);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$OFFSET_Y$", "" + offsetY);
        programGetEntropyMap = context.createProgram(programStringGetEntropyMap).build();

        // ---- Fill buffers ----
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

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

        // Weighted mean NRMSE map
        argn = 0;
        kernelGetNrmseMap.setArg(argn++, clRefPixels);
        kernelGetNrmseMap.setArg(argn++, clLocalMeans);
        kernelGetNrmseMap.setArg(argn++, clNrmseMap);
        kernelGetNrmseMap.setArg(argn++, clMaeMap);

        // Weighted mean SSIM map
        argn = 0;
        kernelGetSsimMap.setArg(argn++, clRefPixels);
        kernelGetSsimMap.setArg(argn++, clLocalMeans);
        kernelGetSsimMap.setArg(argn++, clSsimMap);

        // Hu map
        argn = 0;
        kernelGetHuMap.setArg(argn++, clRefPixels);
        kernelGetHuMap.setArg(argn++, clLocalMeans);
        kernelGetHuMap.setArg(argn++, clHuMap);

        // Entropy map
        argn = 0;
        kernelGetEntropyMap.setArg(argn++, clRefPixels8Bit);
        kernelGetEntropyMap.setArg(argn++, clLocalMeans);
        kernelGetEntropyMap.setArg(argn++, clEntropyMap);

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
                queue.put2DRangeKernel(kernelGetLocalMeans, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
            }
        }

        //queue.put1DRangeKernel(kernelGetLocalMeans, 0, w*h, 0);
        queue.finish();

        // ---- Read the local means map back from the GPU ----
        queue.putReadBuffer(clLocalMeans, true);
        for (int v = 0; v < localMeans.length; v++) {
            localMeans[v] = clLocalMeans.getBuffer().get(v);
        }
        queue.finish();

        kernelGetLocalMeans.release();
        programGetLocalMeans.release();

        //IJ.log("Done!");
        //IJ.log("--------");

        // ---- Calculate weighted mean Pearson's map ----
        queue.putWriteBuffer(clPearsonMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Pearson's correlations... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetPearsonMap, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
            }
        }

        //queue.put2DRangeKernel(kernelGetPearsonMap, 0, 0, w, h, 0,0);
        queue.finish();

        // ---- Read the Pearson's map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clPearsonMap, true);
        for (int a = 0; a<h; a++) {
            for(int b=0; b<w; b++) {
                pearsonMap[a*w+b] = clPearsonMap.getBuffer().get(a*w+b) / sizeWithoutBorders;
                queue.finish();
            }
        }

        kernelGetPearsonMap.release();
        programGetPearsonMap.release();
        clPearsonMap.release();

        // ---- Calculate weighted mean NRMSE map ----
        queue.putWriteBuffer(clNrmseMap, false);

        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating NRMSE and MAE... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetNrmseMap, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
            }
        }

        //queue.put2DRangeKernel(kernelGetNrmseMap, 0, 0, w, h, 0,0);
        queue.finish();

        // ---- Read the NRMSE and MAE maps back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clNrmseMap, true);
        for (int c=0; c<h; c++) {
            for(int d=0; d<w; d++) {
                nrmseMap[c*w+d] = clNrmseMap.getBuffer().get(c*w+d) / sizeWithoutBorders;
                queue.finish();
            }
        }

        queue.putReadBuffer(clMaeMap, true);
        for (int e=0; e<h; e++) {
            for (int f=0; f<w; f++) {
                maeMap[e*w+f] = clMaeMap.getBuffer().get(e*w+f) / sizeWithoutBorders;
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
                queue.put2DRangeKernel(kernelGetSsimMap, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
            }
        }

        //queue.put2DRangeKernel(kernelGetSsimMap, 0, 0, w, h, 0,0);
        queue.finish();

        // ---- Read the SSIM map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clSsimMap, true);
        for (int g=0; g<h; g++) {
            for (int i=0; i<w; i++) {
                ssimMap[g*w+i] = clSsimMap.getBuffer().get(g*w+i) / sizeWithoutBorders;
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
                queue.put2DRangeKernel(kernelGetHuMap, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
            }
        }
        queue.finish();

        // ---- Read the Hu map back from the GPU (and finish the mean calculation simultaneously) ----
        queue.putReadBuffer(clHuMap, true);
        for (int j=0; j<h; j++) {
            for (int k=0; k<w; k++) {
                huMap[j*w+k] = clHuMap.getBuffer().get(j*w+k) / sizeWithoutBorders;
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
                queue.put2DRangeKernel(kernelGetEntropyMap, nXB*64+offsetX, nYB*64+offsetY, xWorkSize, yWorkSize, 0, 0);
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
        // ---- Cleanup all resources associated with this context ----
        IJ.log("Cleaning up resources...");
        context.release();
        IJ.log("Done!");
        IJ.log("--------");

        // ---- Display results ----
        IJ.log("Preparing results for display...");

        // Pearson's map (normalized to [0,1])
        float[] pearsonMinMax = findMinMax(pearsonMap, w, h, offsetX, offsetY);
        float[] pearsonMapNorm = normalize(pearsonMap, w, h, offsetX, offsetY, pearsonMinMax);
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMapNorm);
        ImagePlus imp1 = new ImagePlus("Pearson's Map", fp1);
        imp1.show();

        // NRMSE map (normalized to [0,1])
        float[] nrmseMinMax = findMinMax(nrmseMap, w, h, offsetX, offsetY);
        float[] nrmseMapNorm = normalize(nrmseMap, w, h, offsetX, offsetY, nrmseMinMax);
        FloatProcessor fp2 = new FloatProcessor(w, h, nrmseMapNorm);
        ImagePlus imp2 = new ImagePlus("NRMSE Map", fp2);
        imp2.show();

        // MAE map (normalized to [0,1])
        float[] maeMinMax = findMinMax(maeMap, w, h, offsetX, offsetY);
        float[] maeMapNorm = normalize(maeMap, w, h, offsetX, offsetY, maeMinMax);
        FloatProcessor fp3 = new FloatProcessor(w, h, maeMapNorm);
        ImagePlus imp3 = new ImagePlus("MAE Map", fp3);
        imp3.show();

        // SSIM map (normalized to [0,1])
        float[] ssimMinMax = findMinMax(ssimMap, w, h, offsetX, offsetY);
        float[] ssimMapNorm = normalize(ssimMap, w, h, offsetX, offsetY, ssimMinMax);
        FloatProcessor fp4 = new FloatProcessor(w, h, ssimMapNorm);
        ImagePlus imp4 = new ImagePlus("SSIM Map", fp4);
        imp4.show();

        // Hu map (normalized to [0,1])
        float[] huMinMax = findMinMax(huMap, w, h, offsetX, offsetY);
        float[] huMapNorm = normalize(huMap, w, h, offsetX, offsetY, huMinMax);
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

    public static float[] normalize(float[] rawPixels, int w, int h, int offsetX, int offsetY, float[] minMax){
        float min = minMax[0];

        float denominator = minMax[1] - min;
        float[] normalizedPixels = new float[w*h];

        for(int j=offsetY; j<h-offsetY; j++) {
            for (int i=offsetX; i<w-offsetX; i++) {
                normalizedPixels[j*w+i] = (rawPixels[j*w+i]-min)/denominator;
            }
        }
        return normalizedPixels;
    }
}