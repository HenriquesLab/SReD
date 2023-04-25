import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.UserFunction;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.awt.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.*;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;

public class GlobalRedundancy implements Runnable, UserFunction{

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;
    static private CLCommandQueue queue;
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetDiffStdMap, programGetNrmseMap, programGetSsimMap,
            programGetHuMap, programGetEntropyMap, programGetPhaseCorrelationMap, programGetHausdorffMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetDiffStdMap, kernelGetNrmseMap, kernelGetSsimMap,
            kernelGetHuMap, kernelGetEntropyMap, kernelGetPhaseCorrelationMap, kernelGetHausdorffMap;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clDiffStdMap, clNrmseMap, clMaeMap, clPsnrMap,
            clSsimMap, clLuminanceMap, clContrastMap, clStructureMap, clHuMap, clEntropyMap, clPhaseCorrelationMap, clHausdorffMap;

    private CLBuffer<IntBuffer> clUniqueStdCoords;


    // -------------------------- //
    // ---- Image parameters ---- //
    // -------------------------- //

    public float[] refPixels, localMeans, localStds, pearsonMap, diffStdMap, nrmseMap, maeMap, psnrMap, ssimMap, luminanceMap,
            contrastMap, structureMap, huMap, entropyMap, phaseCorrelationMap, hausdorffMap;
    public int w, h, wh, bW, bH, patchSize, bRW, bRH, sizeWithoutBorders, speedUp, useGAT, rotInv, scaleFactor, level, w0, h0;
    public float EPSILON;

    public GlobalRedundancy(float[] refPixels, int w, int h, int bW, int bH, float EPSILON, CLContext context,
                            CLCommandQueue queue, int speedUp, int useGAT, int rotInv, int scaleFactor, int level, int w0, int h0){
        this.refPixels = refPixels;
        this.w = w;
        this.h = h;
        wh = w * h;
        this.bW = bW;
        this.bH = bH;
        bRW = bW/2;
        bRH = bH/2;
        sizeWithoutBorders = (w - bRW * 2) * (h - bRH * 2);
        patchSize = (2*bRW+1) * (2*bRW+1) - (int) ceil((sqrt(2)*bRW)*(sqrt(2)*bRW)); // Number of pixels in circular patch

        this.EPSILON = EPSILON;
        this.context = context;
        this.queue = queue;
        this.speedUp = speedUp;
        this.useGAT = useGAT;
        this.rotInv = rotInv;
        this.scaleFactor = scaleFactor;
        this.level = level;
        this.w0 = w0;
        this.h0 = h0;
        localMeans = new float[wh];
        localStds = new float[wh];
        pearsonMap = new float[wh];
        diffStdMap = new float[wh];
        nrmseMap = new float[wh];
        maeMap = new float[wh];
        psnrMap = new float[wh];
        ssimMap = new float[wh];
        luminanceMap = new float[wh];
        contrastMap = new float[wh];
        structureMap = new float[wh];
        huMap = new float[wh];
        entropyMap = new float[wh];
        phaseCorrelationMap = new float[wh];
        hausdorffMap = new float[wh];
    }

    @Override
    public void run(){

        IJ.log("Calculating level "+level);

        // --------------------------------------------------------------------------- //
        // ---- Stabilize noise variance using the Generalized Anscombe transform ---- //
        // --------------------------------------------------------------------------- //

        if(useGAT == 1) {
            // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
            GATMinimizer minimizer = new GATMinimizer(refPixels, w, h, 0, 100, 0);
            minimizer.run();

            // Get gain, sigma and offset from minimizer and transform pixel values
            refPixels = TransformImageByVST_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);
        }


        // ------------------------- //
        // ---- Normalize image ---- //
        // ------------------------- //

        float minMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 1, 2);


        // ------------------------------------------------------------------------- //
        // ---- Estimate noise standard deviation (used for weight calculation) ---- //
        // ------------------------------------------------------------------------- //

        float noiseVar = estimateNoiseVar(refPixels, w, h);

        // -------------------------------------------------------- //
        // ---- Write input image (variance-stabilized) to GPU ---- //
        // -------------------------------------------------------- //

        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        fillBufferWithFloatArray(clRefPixels, refPixels);
        queue.putWriteBuffer(clRefPixels, false);


        // ------------------------------------------------------------ //
        // ---- Calculate local means and standard deviations maps ---- //
        // ------------------------------------------------------------ //

        // Create OpenCL program
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BW$", "" + bW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BH$", "" + bH);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRW$", "" + bRW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRH$", "" + bRH);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        // Create, fill and write OpenCL buffers
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalMeans, localMeans);
        queue.putWriteBuffer(clLocalMeans, false);

        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalStds, localStds);
        queue.putWriteBuffer(clLocalStds, false);

        // Create OpenCL kernel and set kernel arguments
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");

        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);
        kernelGetLocalMeans.setArg(argn++, clLocalStds);

        // Calculate local means and standard deviations map in the GPU
        int nXBlocks = w/64 + ((w%64==0)?0:1);
        int nYBlocks = h/64 + ((h%64==0)?0:1);
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating local means... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetLocalMeans, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the local means map back from the GPU
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                localMeans[y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Read the local stds map back from the GPU
        queue.putReadBuffer(clLocalStds, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                localStds[y*w+x] = clLocalStds.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release GPU resources
        kernelGetLocalMeans.release(); // TODO: MAKES SENSE TO RELEASE KERNELS HERE? WILL THEY BE USED AGAIN?
        programGetLocalMeans.release();


        // ----------------------------------------------------------------------------------------------- //
        // ---- Get array of unique StdDev values and a set of coordinates for each, and write to GPU ---- //
        // ----------------------------------------------------------------------------------------------- //

        float[] stdUnique = getUniqueValues(localStds, w, h, bRW, bRH);
        int[] stdUniqueCoords = getUniqueValueCoordinates(stdUnique, localStds, w, h, bRW, bRH);
        int nUnique = stdUnique.length;

        clUniqueStdCoords = context.createIntBuffer(stdUniqueCoords.length, READ_ONLY);
        fillBufferWithIntArray(clUniqueStdCoords, stdUniqueCoords);
        queue.putWriteBuffer(clUniqueStdCoords, false);


        // -------------------------------------------------------------------- //
        // ---- Calculate weighted mean absolute difference of StdDevs map ---- //
        // -------------------------------------------------------------------- //

        if(rotInv == 1){

            // Create OpenCL program
            String programStringGetDiffStdMap = getResourceAsString(RedundancyMap_.class, "kernelGetDiffStdMap.cl");
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$WIDTH$", "" + w);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$HEIGHT$", "" + h);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BW$", "" + bW);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BH$", "" + bH);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$PATCH_SIZE$", "" + patchSize);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRW$", "" + bRW);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRH$", "" + bRH);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$FILTERPARAM$", "" + noiseVar);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$EPSILON$", "" + EPSILON);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$NUNIQUE$", "" + nUnique);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$SPEEDUP$", "" + speedUp);
            programGetDiffStdMap = context.createProgram(programStringGetDiffStdMap).build();

            // Create, fill and write OpenCL buffers
            clDiffStdMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clDiffStdMap, diffStdMap);
            queue.putWriteBuffer(clDiffStdMap, false);

            // Create OpenCL kernel and set kernel args
            kernelGetDiffStdMap = programGetDiffStdMap.createCLKernel("kernelGetDiffStdMap");

            argn = 0;
            kernelGetDiffStdMap.setArg(argn++, clRefPixels);
            kernelGetDiffStdMap.setArg(argn++, clLocalMeans);
            kernelGetDiffStdMap.setArg(argn++, clLocalStds);
            kernelGetDiffStdMap.setArg(argn++, clUniqueStdCoords);
            kernelGetDiffStdMap.setArg(argn++, clDiffStdMap);

            // Calculate weighted mean absolute difference of standard deviations map in the GPU
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, h - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, w - nXB * 64);
                    showStatus("Calculating redundancy... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                    queue.put2DRangeKernel(kernelGetDiffStdMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                    queue.finish();
                }
            }

            // Read the weighted mean absolute difference of standard deviations map back from the GPU (and finish the mean calculation simultaneously)
            queue.putReadBuffer(clDiffStdMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    diffStdMap[y * w + x] = clDiffStdMap.getBuffer().get(y * w + x) / sizeWithoutBorders;
                    queue.finish();
                }
            }

            // Release GPU resources
            kernelGetDiffStdMap.release();
            programGetDiffStdMap.release();
            clDiffStdMap.release();

            // Invert values (because so far we have inverse frequencies)
            float[] diffStdMinMax = findMinMax(diffStdMap, w, h, bRW, bRH);
            diffStdMap = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);
            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    diffStdMap[j*w+i] = 1.0f - diffStdMap[j*w+i];
                }
            }
            diffStdMap = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);

            // Filter out regions with low noise variance
            float[] localVars = new float[wh];
            float noiseMeanVar = 0.0f;
            int counter = 0;
            float value = 0.0f;
            for(int j=0; j<h; j++){
                for(int i=0; i<w; i++){
                    value = localStds[j*w+i]*localStds[j*w+i];
                    localVars[j*w+i] = value;
                    noiseMeanVar += value;
                    counter++;
                }
            }
            noiseMeanVar /= counter;

            for(int j=0; j<h; j++){
                for(int i=0; i<w; i++){
                    if(localVars[j*w+i]<noiseMeanVar){
                        diffStdMap[j*w+i] = 0.0f;
                    }
                }
            }

        }


        // ----------------------------------------------- //
        // ---- Calculate weighted mean Pearson's map ---- //
        // ----------------------------------------------- //

        if(rotInv == 0){

            // Create OpenCL program
            String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetPearsonMap.cl");
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BW$", "" + bW);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BH$", "" + bH);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRW$", "" + bRW);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRH$", "" + bRH);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTERPARAM$", "" + noiseVar);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$EPSILON$", "" + EPSILON);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$NUNIQUE$", "" + nUnique);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$SPEEDUP$", "" + speedUp);
            programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

            // Create, fill and write OpenCL buffers
            clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clPearsonMap, pearsonMap);
            queue.putWriteBuffer(clPearsonMap, false);

            // Create OpenCL kernel and set kernel args
            kernelGetPearsonMap = programGetPearsonMap.createCLKernel("kernelGetPearsonMap");

            argn = 0;
            kernelGetPearsonMap.setArg(argn++, clRefPixels);
            kernelGetPearsonMap.setArg(argn++, clLocalMeans);
            kernelGetPearsonMap.setArg(argn++, clLocalStds);
            kernelGetPearsonMap.setArg(argn++, clUniqueStdCoords);
            kernelGetPearsonMap.setArg(argn++, clPearsonMap);

            // Calculate weighted mean Pearson's map in the GPU
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, h - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, w - nXB * 64);
                    showStatus("Calculating redundancy... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                    queue.put2DRangeKernel(kernelGetPearsonMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                    queue.finish();
                }
            }

            // Read the weighted mean  Pearson's map back from the GPU (and finish the mean calculation simultaneously)
            queue.putReadBuffer(clPearsonMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    pearsonMap[y * w + x] = clPearsonMap.getBuffer().get(y * w + x) / sizeWithoutBorders;
                    queue.finish();
                }
            }

            // Release GPU resources
            kernelGetPearsonMap.release();
            programGetPearsonMap.release();
            clPearsonMap.release();

            // Filter out regions with low noise variance
            float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
            pearsonMap = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);

            float[] localVars = new float[wh];
            float noiseMeanVar = 0.0f;
            int counter = 0;
            float value;
            for(int j=0; j<h; j++){
                for(int i=0; i<w; i++){
                    value = localStds[j*w+i]*localStds[j*w+i];
                    localVars[j*w+i] = value;
                    if(i<bRW || i>=w-bRW || j<bRH || j>=h-bRH){
                        continue;
                    }
                    noiseMeanVar += value;
                    counter++;
                }
            }
            noiseMeanVar /= counter;

            for(int j=0; j<h; j++){
                for(int i=0; i<w; i++){
                    if(localVars[j*w+i]<noiseMeanVar){
                        pearsonMap[j*w+i] = 0.0f;
                    }
                }
            }
        }


/*
        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            pearsonMap = remapPixels(pearsonMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }

        // ---- Calculate weighted mean NRMSE, MAE and PSNR maps ----
        // Create OpenCL program
        String programStringGetNrmseMap = getResourceAsString(RedundancyMap_.class, "kernelGetNrmseMap.cl");
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$WIDTH$", "" + w);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$HEIGHT$", "" + h);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BW$", "" + bW);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BH$", "" + bH);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BRW$", "" + bRW);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$BRH$", "" + bRH);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$EPSILON$", "" + EPSILON);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$NUNIQUE$", "" + nUnique);
        programStringGetNrmseMap = replaceFirst(programStringGetNrmseMap, "$SPEEDUP$", "" + speedUp);
        programGetNrmseMap = context.createProgram(programStringGetNrmseMap).build();

        // Create, fill and write buffers
        clNrmseMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clNrmseMap, nrmseMap);
        queue.putWriteBuffer(clNrmseMap, false);

        clMaeMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clMaeMap, maeMap);
        queue.putWriteBuffer(clMaeMap, false);

        clPsnrMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPsnrMap, psnrMap);
        queue.putWriteBuffer(clPsnrMap, false);

        // Create kernel and set kernel args
        kernelGetNrmseMap = programGetNrmseMap.createCLKernel("kernelGetNrmseMap");

        argn = 0;
        kernelGetNrmseMap.setArg(argn++, clRefPixels);
        kernelGetNrmseMap.setArg(argn++, clLocalMeans);
        kernelGetNrmseMap.setArg(argn++, clLocalStds);
        kernelGetNrmseMap.setArg(argn++, clUniqueStdCoords);
        kernelGetNrmseMap.setArg(argn++, clNrmseMap);
        kernelGetNrmseMap.setArg(argn++, clMaeMap);
        kernelGetNrmseMap.setArg(argn++, clPsnrMap);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating NRMSE and MAE... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetNrmseMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the NRMSE, MAE and PSNR maps back from the GPU (and finish the mean calculation simultaneously)
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

        queue.putReadBuffer(clPsnrMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                psnrMap[y*w+x] = clPsnrMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetNrmseMap.release();
        programGetNrmseMap.release();
        clNrmseMap.release();
        clMaeMap.release();
        clPsnrMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            nrmseMap = remapPixels(nrmseMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
            maeMap = remapPixels(maeMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
            psnrMap = remapPixels(psnrMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }


 */
        /*
        // ---- Calculate weighted mean SSIM map ----
        // Create OpenCL program
        String programStringGetSsimMap = getResourceAsString(RedundancyMap_.class, "kernelGetSsimMap.cl");
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$WIDTH$", "" + w);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$HEIGHT$", "" + h);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BW$", "" + bW);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BH$", "" + bH);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BRW$", "" + bRW);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$BRH$", "" + bRH);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$EPSILON$", "" + EPSILON);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$NUNIQUE$", "" + nUnique);
        programStringGetSsimMap = replaceFirst(programStringGetSsimMap, "$SPEEDUP$", "" + speedUp);
        programGetSsimMap = context.createProgram(programStringGetSsimMap).build();

        // Create, fill and write buffers
        clSsimMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clSsimMap, ssimMap);
        queue.putWriteBuffer(clSsimMap, false);

        clLuminanceMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLuminanceMap, luminanceMap);
        queue.putWriteBuffer(clLuminanceMap, false);

        clContrastMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clContrastMap, contrastMap);
        queue.putWriteBuffer(clContrastMap, false);

        clStructureMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clStructureMap, structureMap);
        queue.putWriteBuffer(clStructureMap, false);

        // Create kernel and set kernel args
        kernelGetSsimMap = programGetSsimMap.createCLKernel("kernelGetSsimMap");

        argn = 0;
        kernelGetSsimMap.setArg(argn++, clRefPixels);
        kernelGetSsimMap.setArg(argn++, clLocalMeans);
        kernelGetSsimMap.setArg(argn++, clLocalStds);
        kernelGetSsimMap.setArg(argn++, clUniqueStdCoords);
        kernelGetSsimMap.setArg(argn++, clSsimMap);
        kernelGetSsimMap.setArg(argn++, clLuminanceMap);
        kernelGetSsimMap.setArg(argn++, clContrastMap);
        kernelGetSsimMap.setArg(argn++, clStructureMap);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating SSIM... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetSsimMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the SSIM map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clSsimMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                ssimMap[y*w+x] = clSsimMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        queue.putReadBuffer(clLuminanceMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                luminanceMap[y*w+x] = clLuminanceMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        queue.putReadBuffer(clContrastMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                contrastMap[y*w+x] = clContrastMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        queue.putReadBuffer(clStructureMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                structureMap[y*w+x] = clStructureMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }
        // Release resources
        kernelGetSsimMap.release();
        programGetSsimMap.release();
        clSsimMap.release();
        clLuminanceMap.release();
        clContrastMap.release();
        clStructureMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            ssimMap = remapPixels(ssimMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }
/*
        // ---- Calculate Hu map ----
        // Create OpenCL program
        String programStringGetHuMap = getResourceAsString(RedundancyMap_.class, "kernelGetHuMap.cl");
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$WIDTH$", "" + w);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$HEIGHT$", "" + h);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BW$", "" + bW);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BH$", "" + bH);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BRW$", "" + bRW);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$BRH$", "" + bRH);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$EPSILON$", "" + EPSILON);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$NUNIQUE$", "" + nUnique);
        programStringGetHuMap = replaceFirst(programStringGetHuMap, "$SPEEDUP$", "" + speedUp);
        programGetHuMap = context.createProgram(programStringGetHuMap).build();

        // Create, fill and write buffers
        clHuMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clHuMap, huMap);
        queue.putWriteBuffer(clHuMap, false);

        // Create kernel and set kernel args
        kernelGetHuMap = programGetHuMap.createCLKernel("kernelGetHuMap");

        argn = 0;
        kernelGetHuMap.setArg(argn++, clRefPixels);
        kernelGetHuMap.setArg(argn++, clLocalMeans);
        kernelGetHuMap.setArg(argn++, clLocalStds);
        kernelGetHuMap.setArg(argn++, clUniqueStdCoords);
        kernelGetHuMap.setArg(argn++, clHuMap);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Hu... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetHuMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the Hu map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clHuMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                huMap[y*w+x] = clHuMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetHuMap.release();
        programGetHuMap.release();
        clHuMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            huMap = remapPixels(huMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }

        // ---- Calculate entropy map ----
        // Create OpenCL program
        String programStringGetEntropyMap = getResourceAsString(RedundancyMap_.class, "kernelGetEntropyMap.cl");
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$WIDTH$", "" + w);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$HEIGHT$", "" + h);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BW$", "" + bW);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BH$", "" + bH);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BRW$", "" + bRW);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$BRH$", "" + bRH);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap,"$EPSILON$", "" + EPSILON);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$NUNIQUE$", "" + nUnique);
        programStringGetEntropyMap = replaceFirst(programStringGetEntropyMap, "$SPEEDUP$", "" + speedUp);
        programGetEntropyMap = context.createProgram(programStringGetEntropyMap).build();

        // Create, fill and write buffers
        clEntropyMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clEntropyMap, entropyMap);
        queue.putWriteBuffer(clEntropyMap, false);

        // Create kernel and set kernel args
        kernelGetEntropyMap = programGetEntropyMap.createCLKernel("kernelGetEntropyMap");

        argn = 0;
        kernelGetEntropyMap.setArg(argn++, clRefPixels);
        kernelGetEntropyMap.setArg(argn++, clLocalMeans);
        kernelGetEntropyMap.setArg(argn++, clLocalStds);
        kernelGetEntropyMap.setArg(argn++, clUniqueStdCoords);
        kernelGetEntropyMap.setArg(argn++, clEntropyMap);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating entropy... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetEntropyMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the entropy map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clEntropyMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                entropyMap[y*w+x] = clEntropyMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetEntropyMap.release();
        programGetEntropyMap.release();
        clEntropyMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            entropyMap = remapPixels(entropyMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }

/*
        // ---- Calculate Phase correlation map ----
        // Create OpenCL program
        String programStringGetPhaseCorrelationMap = getResourceAsString(RedundancyMap_.class, "kernelGetPhaseCorrelationMap.cl");
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$WIDTH$", "" + w);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$HEIGHT$", "" + h);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BW$", "" + bW);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BH$", "" + bH);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BRW$", "" + bRW);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$BRH$", "" + bRH);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$EPSILON$", "" + EPSILON);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$NUNIQUE$", "" + nUnique);
        programStringGetPhaseCorrelationMap = replaceFirst(programStringGetPhaseCorrelationMap, "$SPEEDUP$", "" + speedUp);
        programGetPhaseCorrelationMap = context.createProgram(programStringGetPhaseCorrelationMap).build();

        // Create, fill and write buffers
        clPhaseCorrelationMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPhaseCorrelationMap, phaseCorrelationMap);
        queue.putWriteBuffer(clPhaseCorrelationMap, false);

        // Create kernel and set kernel args
        kernelGetPhaseCorrelationMap = programGetPhaseCorrelationMap.createCLKernel("kernelGetPhaseCorrelationMap");

        argn = 0;
        kernelGetPhaseCorrelationMap.setArg(argn++, clRefPixels);
        kernelGetPhaseCorrelationMap.setArg(argn++, clLocalMeans);
        kernelGetPhaseCorrelationMap.setArg(argn++, clLocalStds);
        kernelGetPhaseCorrelationMap.setArg(argn++, clUniqueStdCoords);
        kernelGetPhaseCorrelationMap.setArg(argn++, clPhaseCorrelationMap);

        for(int nYB=0; nYB<nYBlocks; nYB++){
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++){
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Phase correlations... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetPhaseCorrelationMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the Phase correlations map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clPhaseCorrelationMap, true);
        for(int y=0; y<h; y++){
            for (int x=0; x<w; x++){
                phaseCorrelationMap[y*w+x] = clPhaseCorrelationMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetPhaseCorrelationMap.release();
        programGetPhaseCorrelationMap.release();
        clPhaseCorrelationMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            phaseCorrelationMap = remapPixels(phaseCorrelationMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }

        // ---- Calculate weighted mean Hausdorff distance map ----
        // Create OpenCL program
        String programStringGetHausdorffMap = getResourceAsString(RedundancyMap_.class, "kernelGetHausdorffMap.cl");
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$WIDTH$", "" + w);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$HEIGHT$", "" + h);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$BW$", "" + bW);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$BH$", "" + bH);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$BRW$", "" + bRW);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$BRH$", "" + bRH);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$FILTERPARAM$", "" + noiseStdDev);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$EPSILON$", "" + EPSILON);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$NUNIQUE$", "" + nUnique);
        programStringGetHausdorffMap = replaceFirst(programStringGetHausdorffMap, "$SPEEDUP$", "" + speedUp);
        programGetHausdorffMap = context.createProgram(programStringGetHausdorffMap).build();

        // Create, fill and write buffers
        clHausdorffMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clHausdorffMap, hausdorffMap);
        queue.putWriteBuffer(clHausdorffMap, false);

        // Create kernel and set kernel args
        kernelGetHausdorffMap = programGetHausdorffMap.createCLKernel("kernelGetHausdorffMap");

        argn = 0;
        kernelGetHausdorffMap.setArg(argn++, clRefPixels);
        kernelGetHausdorffMap.setArg(argn++, clLocalMeans);
        kernelGetHausdorffMap.setArg(argn++, clLocalStds);
        kernelGetHausdorffMap.setArg(argn++, clUniqueStdCoords);
        kernelGetHausdorffMap.setArg(argn++, clHausdorffMap);
        kernelGetHausdorffMap.setArg(argn++, clWeightSum);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Hausdorff distances... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetHausdorffMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the weight sum map back from the GPU
        queue.putReadBuffer(clWeightSum, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                weightSum[y*w+x] = clWeightSum.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Read the Hausdorff map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clHausdorffMap, true);
        for (int y = 0; y<h; y++) {
            for(int x=0; x<w; x++) {
                hausdorffMap[y*w+x] = clHausdorffMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetHausdorffMap.release();
        programGetHausdorffMap.release();
        clHausdorffMap.release();

        */

        IJ.log("Done!");
        IJ.log("--------");


        // ------------------------- //
        // ---- Display results ---- //
        // ------------------------- //

        IJ.log("Preparing results for display...");

        // Absolute difference of standard deviations map (normalized to [0,1])
        float[] diffStdMinMax = findMinMax(diffStdMap, w, h, bRW, bRH);
        float[] diffStdMapNorm = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);
        FloatProcessor fp1 = new FloatProcessor(w, h, diffStdMapNorm);

        // Pearson's map (normalized to [0,1])
        float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
        float[] pearsonMapNorm = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);
        FloatProcessor fp2 = new FloatProcessor(w, h, pearsonMapNorm);

        /*
        // NRMSE map (normalized to [0,1])
        float[] nrmseMinMax = findMinMax(nrmseMap, w, h, bRW, bRH);
        float[] nrmseMapNorm = normalize(nrmseMap, w, h, bRW, bRH, nrmseMinMax, 0, 0);
        FloatProcessor fp2 = new FloatProcessor(w, h, nrmseMapNorm);

        // MAE map (normalized to [0,1])
        float[] maeMinMax = findMinMax(maeMap, w, h, bRW, bRH);
        float[] maeMapNorm = normalize(maeMap, w, h, bRW, bRH, maeMinMax, 0, 0);
        FloatProcessor fp3 = new FloatProcessor(w, h, maeMapNorm);

        // PSNR map (normalized to [0,1]
        float[] psnrMinMax = findMinMax(psnrMap, w, h, bRW, bRH);
        float[] psnrMapNorm = normalize(psnrMap, w, h, bRW, bRH, psnrMinMax, 0, 0);
        FloatProcessor fp4 = new FloatProcessor(w, h, psnrMapNorm);

*/
/*
        // SSIM map (normalized to [0,1])
        float[] ssimMinMax = findMinMax(ssimMap, w, h, bRW, bRH);
        float[] ssimMapNorm = normalize(ssimMap, w, h, bRW, bRH, ssimMinMax, 0, 0);
        FloatProcessor fp5 = new FloatProcessor(w, h, ssimMapNorm);

        float[] luminanceMinMax = findMinMax(luminanceMap, w, h, bRW, bRH);
        float[] luminanceMapNorm = normalize(luminanceMap, w, h, bRW, bRH, luminanceMinMax, 0, 0);
        FloatProcessor fp11 = new FloatProcessor(w, h, luminanceMapNorm);

        float[] contrastMinMax = findMinMax(contrastMap, w, h, bRW, bRH);
        float[] contrastMapNorm = normalize(contrastMap, w, h, bRW, bRH, contrastMinMax, 0, 0);
        FloatProcessor fp12 = new FloatProcessor(w, h, contrastMapNorm);

        float[] structureMinMax = findMinMax(structureMap, w, h, bRW, bRH);
        float[] structureMapNorm = normalize(structureMap, w, h, bRW, bRH, structureMinMax, 0, 0);
        FloatProcessor fp13 = new FloatProcessor(w, h, structureMapNorm);
        */
/*
        // Hu map (normalized to [0,1])
        float[] huMinMax = findMinMax(huMap, w, h, bRW, bRH);
        float[] huMapNorm = normalize(huMap, w, h, bRW, bRH, huMinMax, 0, 0);
        FloatProcessor fp6 = new FloatProcessor(w, h, huMapNorm);

        // Entropy map (normalized to [0,1])
        float[] entropyMinMax = findMinMax(entropyMap, w, h, bRW, bRH);
        float[] entropyMapNorm = normalize(entropyMap, w, h, bRW, bRH, entropyMinMax, 0, 0);
        FloatProcessor fp7 = new FloatProcessor(w, h, entropyMapNorm);

        // Phase map (normalized to [0,1])
        //float[] phaseMinMax = findMinMax(phaseCorrelationMap, w, h, bRW, bRH);
        //float[] phaseMapNorm = normalize(phaseCorrelationMap, w, h, bRW, bRH, phaseMinMax, 0, 0);
        //FloatProcessor fp8 = new FloatProcessor(w, h, phaseMapNorm);


        // Hausdorff map (normalized to [0,1])
        float[] hausdorffMinMax = findMinMax(hausdorffMap, w, h, bRW, bRH);
        float[] hausdorffMapNorm = normalize(hausdorffMap, w, h, bRW, bRH, hausdorffMinMax, 0, 0);
        FloatProcessor fp14 = new FloatProcessor(w, h, hausdorffMapNorm);
*/

        // Create image stack holding the redundancy maps and display it
        ImageStack ims = new ImageStack(w, h);
        //FloatProcessor inputImage = new FloatProcessor(w, h, refPixels);
        //ims.addSlice("Variance-stabilised image", inputImage);
        ims.addSlice("Absolute Difference of StdDevs", fp1);
        ims.addSlice("Pearson", fp2);


        FloatProcessor fp3 = new FloatProcessor(w, h, localStds);
        //fp9 = fp9.resize(w0, h0, true).convertToFloatProcessor();
        ims.addSlice("Local stds", fp3);

        //FloatProcessor fp10 = new FloatProcessor(w, h, weightSum);
        //ims.addSlice("Weight sum", fp10);


        // ------------------------------------------- //
        // ---- Scale back to original dimensions ---- // (TODO: CHANGE THIS FOR SINGLE IMAGE INSTEAD OF STACK WHEN STACK ISNT NEEDED ANYMORE)
        // ------------------------------------------- //

        // Resize
        int nSlices = ims.getSize();
        ImageStack imsFinal = new ImageStack(w0, h0, nSlices);
        int newBRW = bRW * scaleFactor;
        int newBRH = bRH * scaleFactor;

        for(int i=1; i<=nSlices;i++) {
            // Get downscaled image without border (so that later we don't interpolate edges)
            FloatProcessor fpDowscaled = ims.getProcessor(i).convertToFloatProcessor();
            fpDowscaled.setRoi(bRW, bRH, w-2*bRW, h-2*bRH);
            FloatProcessor fpCropped = fpDowscaled.crop().convertToFloatProcessor();

            // Upscale crop
            fpCropped.setInterpolationMethod(ImageProcessor.BICUBIC);
            FloatProcessor fpUpscaled = fpCropped.resize(w0-2*newBRW, h0-2*newBRH, true).convertToFloatProcessor();

            // Map upscaled crop to upscaled image with borders
            FloatProcessor fpFinal = new FloatProcessor(w0, h0);
            fpFinal.insert(fpUpscaled, newBRW, newBRH);

            // Normalize while avoiding borders (TODO: OVERIMPOSE BORDERS IN ALL SCALES)
            float[] tempImg = (float[]) fpFinal.getPixels();
            float[] tempMinMax = findMinMax(tempImg, w0, h0, newBRW, newBRH);
            tempImg = normalize(tempImg, w0, h0, newBRW, newBRH, tempMinMax, 0, 0);
            fpFinal = new FloatProcessor(w0, h0, tempImg);
            imsFinal.setProcessor(fpFinal, i);
        }

        ImagePlus impFinal = new ImagePlus("Redundancy Maps (level = " + level + ")", imsFinal);
        impFinal.show();

        // Apply LUT
        IJ.run(impFinal, "mpl-inferno", "");
        //IJ.run(impFinal, "Invert LUT", "");
    }

    @Override
    public double userFunction(double[] doubles, double v) {
        return 0;
    }

    // ---- USER FUNCTIONS ----
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

    public static void fillBufferWithIntArray(CLBuffer<IntBuffer> clBuffer, int[] pixels) {
        IntBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    public static float[] getUniqueValues(float[] inArr, int w, int h, int offsetX, int offsetY){

        // Make a copy of the input array
        float[] tempArr = new float[inArr.length];
        for(int i=0; i<inArr.length; i++){
            tempArr[i] = inArr[i];
        }

        // Sort array
        Arrays.sort(tempArr);

        // Count unique elements
        int count = 0;
        for(int i=0; i<tempArr.length-1; i++){
            if(tempArr[i] != tempArr[i+1]){
                count++;
            }
        }

        // Create array to hold unique elements
        float[] outArr = new float[count];
        int index = 0;
        for(int i=0; i<tempArr.length-1; i++){
            if(tempArr[i] != tempArr[i+1]){
                outArr[index] = tempArr[i];
                index++;
            }
        }

        return outArr;
    }

    public static int[] getUniqueValueCoordinates(float[] uniqueArr, float[] inArr, int w, int h, int bRW, int bRH){
        int[] outArr = new int[uniqueArr.length];
        boolean found = false;
        for(int u=0; u<uniqueArr.length; u++){
            found = false;
            for(int j=bRH; j<h-bRH && !found; j++) {
                for (int i=bRW; i<w-bRW; i++) {
                    if (inArr[j*w+i] == uniqueArr[u]) {
                        outArr[u] = j*w+i;
                        found = true;
                        break;
                    }
                }
            }
        }

        return outArr;
    }

    public static float[] remapPixels(float[] image, int w, int h, float[] localStdsMap, float[] uniqueStds, int[] uniqueStdsCoords, int nUnique, int bRW, int bRH){
        for(int j=bRH; j<h-bRH; j++) {
            for(int i=bRW; i<w-bRW; i++) {
                for(int u=0; u<nUnique; u++) {
                    if (localStdsMap[j*w+i] == uniqueStds[u]){
                        image[j*w+i] = image[uniqueStdsCoords[u]];
                    }
                }
            }
        }

        return image;
    }

    public float[] linspace(float start, float stop, int n){
        float[] output = new float[n];
        // Return empty array if start equals stop, a.k.a., bad user input
        if(start==stop) {
            return output;

        }else if(start > stop){

            float step = abs((start - stop) / n);
            float value = start;
            for (int i = 0; i <= n; i++) {
                value += step;
                output[i] = value;
            }

        }else if(start<stop){

            float step = abs((start - stop) / n);
            float value = stop;
            for (int i=0; i<=n; i++){
                value -= step;
                output[i] = value;
            }
        }

        return output;
    }

    float estimateNoiseVar(float[] inImg, int w, int h){
        //Based on Liu, W. et al. - "A fast noise variance estimation algorithm". doi: 10.1109/PrimeAsia.2011.6075071

        // Define block size based on image dimensions (8x8 if image dimensions are below or equal to 352x288 pixels, 16x16 otherwise)
        int sizeThreshold = 352*288;
        int bL = 0; // Block length
        int bA = 0; // Block area, i.e., total number of pixels in the block

        if(w*h <= sizeThreshold){
            bL = 8;
            bA = bL * bL;
        }else{
            bL = 16;
            bA = bL * bL;
        }

        // Get total number of non-overlapping blocks
        int nXBlocks = w / bL;
        int nYBlocks = h / bL;
        int nBlocks = nXBlocks * nYBlocks;

        // Get block variances
        float[] vars = new float[nBlocks]; // Array to store variances
        float[] block = new float[bA]; // Array to temporarily store the current block's pixels
        int counter0; // Counter for temporary block indexes
        int counter1 = 0; // Counter for variances array indexes

        for(int y=0; y<h-bL; y=y+bL){
            for(int x=0; x<w-bL; x=x+bL){
                counter0 = 0;

                // Get current block
                for(int yy=y; yy<y+bL; yy++){
                    for(int xx=x; xx<x+bL; xx++){
                        block[counter0] = inImg[yy*w+xx];
                        counter0++;
                    }
                }

                // Get current block variance (single-pass)
                double sum = 0;
                double sum2 = 0;
                for(int i=0; i<bA; i++){
                    sum += block[i];
                    sum2 += block[i] * block[i];
                }
                double mean = sum / bA;
                vars[counter1] = (float) abs(sum2 / bA - mean * mean);
                counter1++;
            }
        }

        // Get the 3% lowest variances and calculate their average
        Arrays.sort(vars);
        int num = (int) (0.03f * nBlocks + 1); // Number of blocks corresponding to the 3% chosen
        float avgVar = 0;
        for(int i=0; i<=num; i++){
            avgVar += vars[i];
        }
        avgVar /= num;

        // Calculate noise variance
        float noiseVar = (1.0f + 0.001f * (avgVar - 40.0f)) * avgVar;
        return abs(noiseVar);
    }

}
