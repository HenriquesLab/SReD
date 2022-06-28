//TODO: Filling buffer with a patch writes wrong values. Currently the kernels are reading the reference patch from the image buffer based on the patch position. Try to fix this to use a patch written in a buffer.

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
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

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;

public class PatchRed_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetPatchPearson, programGetPatchNrmse, programGetPatchSsim,
            programGetPatchHu, programGetPatchEntropy, programGetPatchPhaseCorrelation;

    static private CLKernel kernelGetPatchMeans, kernelGetPatchPearson, kernelGetPatchNrmse, kernelGetPatchSsim,
            kernelGetPatchHu, kernelGetPatchEntropy, kernelGetPatchPhaseCorrelation;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPatch, clRefPatchMeanSub, clRefPixels, clLocalMeans, clLocalStds, clPearsonMap,
            clNrmseMap, clMaeMap, clPsnrMap,clSsimMap, clHuMap, clEntropyMap, clPhaseCorrelationMap;

    private CLBuffer<DoubleBuffer> clRefPatchDftReal, clRefPatchDftImag;

    @Override
    public void run(String s) {
        float EPSILON = 0.0000001f;
        // ---- Start timer ----
        long start = System.currentTimeMillis();

        // ---- Get reference image and some parameters ----
        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }
        ImageProcessor ip0 = imp0.getProcessor();
        FloatProcessor fp0 = ip0.convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        float[] refPixelsRaw = (float[]) fp0.getPixels();
        int w = fp0.getWidth();
        int h = fp0.getHeight();
        int wh = w * h;

        // Check if patch is selected
        Roi roi = imp0.getRoi();
        if (roi == null) {
            IJ.error("No ROI selected. Please draw a rectangle and try again.");
            return;
        }

        // Check if patch dimensions are odd and get patch parameters
        Rectangle rect = ip0.getRoi(); // Getting ROI from float processor is not working correctly
        int bx = rect.x; // x-coordinate of the top left corner of the rectangle
        int by = rect.y; // y-coordinate of the top left corner of the rectangle
        int bW = rect.width; // Patch width
        int bH = rect.height; // Patch height
        int patchSize = bW * bH; // Patch area
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)
        int centerX = bx + bRW; // Patch center (x-axis)
        int centerY = by + bRH; // Patch center (y-axis)

        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        // ---- Normalize input image ----
        float minMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 1, 2);

        // Get patch values
        float[] refPatch = new float[patchSize];
        int counter = 0;
        for(int y=centerY-bRH; y<=centerY+bRH; y++) {
            for(int x=centerX-bRW; x<=centerX+bRW; x++) {
            refPatch[counter] = refPixels[y*w+x];
            counter++;
            }
        }

        // Get patch statistics
        float patchStats[] = meanVarStd(refPatch);
        float mean = patchStats[0];
        float var = patchStats[1];
        float std = patchStats[2];

        // Get mean subtracted patch
        float[] refPatchMeanSub = new float[patchSize];
        for(int i=0; i<patchSize; i++) {
            refPatchMeanSub[i] = refPatch[i] - mean;
        }

        // Get patch DFT (not using built-in function just to keep the same function as in the OpenCl kernel
        double[] refPatchDftReal = new double[patchSize];
        double[] refPatchDftImag = new double[patchSize];

        for(int j=0; j<bH; j++){
            for(int i=0; i<bW; i++){
                for(int jj=0; jj<bH; jj++){
                    for(int ii=0; ii<bW; ii++){
                        refPatchDftReal[j*bW+i] += (refPatchMeanSub[jj*bW+ii] * cos(2*PI*((i*i/bW)+(j*jj/bH))))/sqrt(patchSize);
                        refPatchDftImag[j*bW+i] -= (refPatchMeanSub[jj*bW+ii] * sin(2*PI*((i*i/bW)+(j*jj/bH))))/sqrt(patchSize);
                    }
                }
            }
        }

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

        // ---- Create command queue ----
        queue = chosenDevice.createCommandQueue();
        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        IJ.log("Calculating redundancy...please wait...");

        // ---- Local Means ----
        // Create buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create program
        String programStringGetPatchMeans = getResourceAsString(PatchRed_.class, "kernelGetPatchMeans.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Fill buffers
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[w*h];
        fillBufferWithFloatArray(clLocalStds, localStds);

        // Create kernel and set args
        kernelGetPatchMeans = programGetPatchMeans.createCLKernel("kernelGetPatchMeans");

        int argn = 0;
        kernelGetPatchMeans.setArg(argn++, clRefPixels);
        kernelGetPatchMeans.setArg(argn++, clLocalMeans);
        kernelGetPatchMeans.setArg(argn++, clLocalStds);

        // Calculate
        queue.putWriteBuffer(clRefPixels, false);
        queue.putWriteBuffer(clLocalMeans, false);

        showStatus("Calculating local means...");

        queue.put2DRangeKernel(kernelGetPatchMeans, 0, 0, w, h, 0, 0);
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

        // Release memory
        kernelGetPatchMeans.release();
        programGetPatchMeans.release();

        // ---- Create buffers ----
        clRefPatch = context.createFloatBuffer(patchSize, READ_ONLY);
        clRefPatchMeanSub = context.createFloatBuffer(patchSize, READ_ONLY);


        // ---- Pearson correlation ----
        String programStringGetPatchPearson = getResourceAsString(PatchRed_.class, "kernelGetPatchPearson.cl");
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$WIDTH$", "" + w);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$HEIGHT$", "" + h);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_X$", "" + centerX);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_Y$", "" + centerY);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRW$", "" + bRW);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRH$", "" + bRH);
        programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$STD_X$", "" + std);
        programGetPatchPearson = context.createProgram(programStringGetPatchPearson).build();

        // Fill buffers
        fillBufferWithFloatArray(clRefPatch, refPatch);
        fillBufferWithFloatArray(clRefPatchMeanSub, refPatchMeanSub);

        float[] pearsonMap = new float[wh];
        clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPearsonMap, pearsonMap);

        // Create kernel and set args
        kernelGetPatchPearson = programGetPatchPearson.createCLKernel("kernelGetPatchPearson");

        argn = 0;
        kernelGetPatchPearson.setArg(argn++, clRefPixels);
        kernelGetPatchPearson.setArg(argn++, clLocalMeans);
        kernelGetPatchPearson.setArg(argn++, clLocalStds);
        kernelGetPatchPearson.setArg(argn++, clPearsonMap);

        // Calculate Pearson's correlation coefficient ----
        queue.putWriteBuffer(clPearsonMap, false);
        queue.put2DRangeKernel(kernelGetPatchPearson, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read Pearson's coefficients back from the GPU
        queue.putReadBuffer(clPearsonMap, true);
        for (int y = 0; y<h; y++) {
            for(int x=0; x<w; x++) {
                pearsonMap[y*w+x] = clPearsonMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchPearson.release();
        clPearsonMap.release();
        programGetPatchPearson.release();

        // ---- NMRSE and MAE ----
        String programStringGetPatchNrmse = getResourceAsString(PatchRed_.class, "kernelGetPatchNrmse.cl");
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$WIDTH$", "" + w);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$HEIGHT$", "" + h);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$CENTER_X$", "" + centerX);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$CENTER_Y$", "" + centerY);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$BRW$", "" + bRW);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$BRH$", "" + bRH);
        programStringGetPatchNrmse = replaceFirst(programStringGetPatchNrmse, "$EPSILON$", "" + EPSILON);

        programGetPatchNrmse = context.createProgram(programStringGetPatchNrmse).build();

        // Create and fill buffers
        float[] nrmseMap = new float[wh];
        clNrmseMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clNrmseMap, nrmseMap);

        float[] maeMap = new float[wh];
        clMaeMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clMaeMap, maeMap);

        float[] psnrMap = new float[wh];
        clPsnrMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPsnrMap, psnrMap);

        // Create kernel and set args
        kernelGetPatchNrmse = programGetPatchNrmse.createCLKernel("kernelGetPatchNrmse");

        argn = 0;
        kernelGetPatchNrmse.setArg(argn++, clRefPixels);
        kernelGetPatchNrmse.setArg(argn++, clLocalMeans);
        kernelGetPatchNrmse.setArg(argn++, clLocalStds);
        kernelGetPatchNrmse.setArg(argn++, clNrmseMap);
        kernelGetPatchNrmse.setArg(argn++, clMaeMap);
        kernelGetPatchNrmse.setArg(argn++, clPsnrMap);

        // Calculate NMRSE and MAE
        queue.putWriteBuffer(clNrmseMap, false);
        queue.putWriteBuffer(clMaeMap, false);
        queue.put2DRangeKernel(kernelGetPatchNrmse, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the NRMSE and MAE maps back from the GPU
        queue.putReadBuffer(clNrmseMap, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                nrmseMap[y*w+x] = clNrmseMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        queue.putReadBuffer(clMaeMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                maeMap[y*w+x] = clMaeMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        queue.putReadBuffer(clPsnrMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                psnrMap[y*w+x] = clPsnrMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchNrmse.release();
        clNrmseMap.release();
        clMaeMap.release();
        clPsnrMap.release();
        programGetPatchNrmse.release();

        // ---- SSIM ----
        String programStringGetPatchSsim = getResourceAsString(PatchRed_.class, "kernelGetPatchSsim.cl");
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$WIDTH$", "" + w);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$HEIGHT$", "" + h);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$CENTER_X$", "" + centerX);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$CENTER_Y$", "" + centerY);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRW$", "" + bRW);
        programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRH$", "" + bRH);
        programGetPatchSsim = context.createProgram(programStringGetPatchSsim).build();

        // Fill buffers
        float[] ssimMap = new float[wh];
        clSsimMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clSsimMap, ssimMap);

        // Create kernel and set args
        kernelGetPatchSsim = programGetPatchSsim.createCLKernel("kernelGetPatchSsim");

        argn = 0;
        kernelGetPatchSsim.setArg(argn++, clRefPixels);
        kernelGetPatchSsim.setArg(argn++, clLocalMeans);
        kernelGetPatchSsim.setArg(argn++, clLocalStds);
        kernelGetPatchSsim.setArg(argn++, clSsimMap);

        // Calculate SSIM
        queue.putWriteBuffer(clSsimMap, false);
        queue.put2DRangeKernel(kernelGetPatchSsim, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the SSIM map back from the GPU
        queue.putReadBuffer(clSsimMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                ssimMap[y*w+x] = clSsimMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchSsim.release();
        clSsimMap.release();
        programGetPatchSsim.release();

        // ---- Hu invariant ----
        String programStringGetPatchHu = getResourceAsString(PatchRed_.class, "kernelGetPatchHu.cl");
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$WIDTH$", "" + w);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$HEIGHT$", "" + h);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$BW$", "" + bW);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$BH$", "" + bH);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$CENTER_X$", "" + centerX);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$CENTER_Y$", "" + centerY);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$BRW$", "" + bRW);
        programStringGetPatchHu = replaceFirst(programStringGetPatchHu, "$BRH$", "" + bRH);
        programGetPatchHu = context.createProgram(programStringGetPatchHu).build();

        // Fill buffers
        float[] huMap = new float[wh];
        clHuMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clHuMap, huMap);

        // Create kernel and set args
        kernelGetPatchHu = programGetPatchHu.createCLKernel("kernelGetPatchHu");

        argn = 0;
        kernelGetPatchHu.setArg(argn++, clRefPixels);
        kernelGetPatchHu.setArg(argn++, clLocalMeans);
        kernelGetPatchHu.setArg(argn++, clLocalStds);
        kernelGetPatchHu.setArg(argn++, clHuMap);

        // Calculate Hu invariant
        queue.putWriteBuffer(clHuMap, false);
        queue.put2DRangeKernel(kernelGetPatchHu, 0, 0, w, h, 0, 0);
        queue.finish();

        // ---- Read the Hu map back from the GPU ----
        queue.putReadBuffer(clHuMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                huMap[y*w+x] = clHuMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchHu.release();
        clHuMap.release();
        programGetPatchHu.release();

        // ---- Phase correlation ----
        String programStringGetPatchPhaseCorrelation = getResourceAsString(PatchRed_.class, "kernelGetPatchPhaseCorrelation.cl");
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$WIDTH$", "" + w);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$HEIGHT$", "" + h);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$BW$", "" + bW);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$BH$", "" + bH);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$CENTER_X$", "" + centerX);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$CENTER_Y$", "" + centerY);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$BRW$", "" + bRW);
        programStringGetPatchPhaseCorrelation = replaceFirst(programStringGetPatchPhaseCorrelation, "$BRH$", "" + bRH);
        programGetPatchPhaseCorrelation = context.createProgram(programStringGetPatchPhaseCorrelation).build();

        // Create and fill buffers
        //clRefPatchDftReal = context.createDoubleBuffer(patchSize, READ_ONLY);
        //fillBufferWithDoubleArray(clRefPatchDftReal, refPatchDftReal);

        //clRefPatchDftImag = context.createDoubleBuffer(patchSize, READ_ONLY);
        //fillBufferWithDoubleArray(clRefPatchDftImag, refPatchDftImag);

        float[] phaseCorrelationMap = new float[wh];
        clPhaseCorrelationMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPhaseCorrelationMap, phaseCorrelationMap);

        // Create kernel and set args
        kernelGetPatchPhaseCorrelation = programGetPatchPhaseCorrelation.createCLKernel("kernelGetPatchPhaseCorrelation");

        argn = 0;
        //kernelGetPatchPhaseCorrelation.setArg(argn++, clRefPatchDftReal);
        //kernelGetPatchPhaseCorrelation.setArg(argn++, clRefPatchDftImag);
        kernelGetPatchPhaseCorrelation.setArg(argn++, clRefPixels);
        kernelGetPatchPhaseCorrelation.setArg(argn++, clLocalMeans);
        kernelGetPatchPhaseCorrelation.setArg(argn++, clLocalStds);
        kernelGetPatchPhaseCorrelation.setArg(argn++, clPhaseCorrelationMap);

        // Calculate phase correlations
        // ---- Calculate Phase correlation map ----
        queue.putWriteBuffer(clPhaseCorrelationMap, false);
        queue.put2DRangeKernel(kernelGetPatchPhaseCorrelation, 0, 0, w, h, 0, 0);
        queue.finish();

        // ---- Read the Phase correlations map back from the GPU ----
        queue.putReadBuffer(clPhaseCorrelationMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                phaseCorrelationMap[y*w+x] = clPhaseCorrelationMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }
        kernelGetPatchPhaseCorrelation.release();
        clPhaseCorrelationMap.release();
        programGetPatchPhaseCorrelation.release();
        //clRefPatchDftReal.release();
        //clRefPatchDftImag.release();
        // Replicate MATLAB's fftshift function for 2D odd-sized images

        // ---- Entropy ----
        String programStringGetPatchEntropy = getResourceAsString(PatchRed_.class, "kernelGetPatchEntropy.cl");
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$WIDTH$", "" + w);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$HEIGHT$", "" + h);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$CENTER_X$", "" + centerX);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$CENTER_Y$", "" + centerY);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$BRW$", "" + bRW);
        programStringGetPatchEntropy = replaceFirst(programStringGetPatchEntropy, "$BRH$", "" + bRH);
        programGetPatchEntropy = context.createProgram(programStringGetPatchEntropy).build();

        // Fill buffers
        float[] entropyMap = new float[wh];
        clEntropyMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clEntropyMap, entropyMap);

        // Create kernel and set args
        kernelGetPatchEntropy = programGetPatchEntropy.createCLKernel("kernelGetPatchEntropy");

        argn = 0;
        kernelGetPatchEntropy.setArg(argn++, clRefPixels);
        kernelGetPatchEntropy.setArg(argn++, clLocalMeans);
        kernelGetPatchEntropy.setArg(argn++, clEntropyMap);

        // Calculate entropy map
        queue.putWriteBuffer(clEntropyMap, false);
        queue.put2DRangeKernel(kernelGetPatchEntropy, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the entropy map back from the GPU
        queue.putReadBuffer(clEntropyMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                entropyMap[y*w+x] = clEntropyMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        kernelGetPatchEntropy.release();
        clEntropyMap.release();
        programGetPatchEntropy.release();

        IJ.log("Done!");
        IJ.log("--------");

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
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMapNorm);
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

        // PSNR map (normalized to [0,1])
        float[] psnrMinMax = findMinMax(psnrMap, w, h, bRW, bRH);
        float[] psnrMapNorm = normalize(psnrMap, w, h, bRW, bRH, psnrMinMax, 0, 0);
        FloatProcessor fp4 = new FloatProcessor(w, h, psnrMapNorm);
        ImagePlus imp4 = new ImagePlus("PSNR Map", fp4);
        imp4.show();

        // SSIM map (normalized to [0,1])
        float[] ssimMinMax = findMinMax(ssimMap, w, h, bRW, bRH);
        float[] ssimMapNorm = normalize(ssimMap, w, h, bRW, bRH, ssimMinMax, 0, 0);
        FloatProcessor fp5 = new FloatProcessor(w, h, ssimMap);
        ImagePlus imp5 = new ImagePlus("SSIM Map", fp5);
        imp5.show();

        // Hu map (normalized to [0,1])
        float[] huMinMax = findMinMax(huMap, w, h, bRW, bRH);
        float[] huMapNorm = normalize(huMap, w, h, bRW, bRH, huMinMax, 0, 0);
        FloatProcessor fp6 = new FloatProcessor(w, h, huMapNorm);
        ImagePlus imp6 = new ImagePlus("Hu Map", fp6);
        imp6.show();

        // Entropy map (normalized to [0,1])
        FloatProcessor fp7 = new FloatProcessor(w, h, entropyMap);
        ImagePlus imp7 = new ImagePlus("Entropy Map", fp7);
        imp7.show();

        // Phase map (normalized to [0,1])
        float[] phaseMinMax = findMinMax(phaseCorrelationMap, w, h, bRW, bRH);
        float[] phaseMapNorm = normalize(phaseCorrelationMap, w, h, bRW, bRH, phaseMinMax, 0, 0);
        FloatProcessor fp8 = new FloatProcessor(w, h, phaseMapNorm);
        ImagePlus imp8 = new ImagePlus("Phase Map", fp8);
        imp8.show();

        // ---- Stop timer ----
        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }

    // ---- USER FUNCTIONS ----
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
        double variance = sq_sum / n - mean * mean;

        return new float[]{(float) mean, (float) variance, (float) sqrt(variance)};

    }

    public static void fillBufferWithFloat(CLBuffer<FloatBuffer> clBuffer, float pixel) {
        FloatBuffer buffer = clBuffer.getBuffer();
        buffer.put(pixel);
    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

    public static void fillBufferWithDoubleArray(CLBuffer<DoubleBuffer> clBuffer, double[] pixels) {
        DoubleBuffer buffer = clBuffer.getBuffer();
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

    public static double getInvariant(float[] patch, int w, int h, int p, int q){
        // Get centroid x and y
        double moment_10 = 0.0f;
        double moment_01 = 0.0f;
        double moment_00 = 0.0f;
        for(int j=0; j<h; j++){
            for(int i=0; i<w; i++){
                moment_10 += patch[j*w+i] * i;
                moment_01 += patch[j*w+i] * j;
                moment_00 += patch[j*w+i];
            }
        }

        double centroid_x = moment_10 / (moment_00 + 0.000001f);
        double centroid_y = moment_01 / (moment_00 + 0.000001f);

        // Get mu_pq
        double mu_pq = 0.0f;
        for(int j=0; j<h; j++){
            for(int i=0; i<w; i++){
                mu_pq += patch[j*w+i] + pow(i+1-centroid_x, p) * pow(j+1-centroid_y, q);
            }
        }

        float invariant = (float) (mu_pq / pow(moment_00, (1+(p+q/2))));
        return invariant;
    }
}

