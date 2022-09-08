import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.UserFunction;
import ij.process.FloatProcessor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.min;
import static nanoj.core2.NanoJCL.replaceFirst;

public class GlobalRedundancy implements UserFunction {

    // OpenCL formats
    static private CLContext context;
    static private CLCommandQueue queue;
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetNrmseMap, programGetSsimMap,
            programGetHuMap, programGetEntropyMap, programGetPhaseCorrelationMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetNrmseMap, kernelGetSsimMap,
            kernelGetHuMap, kernelGetEntropyMap, kernelGetPhaseCorrelationMap;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clNrmseMap, clMaeMap, clPsnrMap,
            clSsimMap, clHuMap, clEntropyMap, clPhaseCorrelationMap;

    private CLBuffer<IntBuffer> clUniqueStdCoords;

    // Image parameters
    public float[] refPixels, localMeans, localStds, pearsonMap, nrmseMap, maeMap, psnrMap, ssimMap, huMap, entropyMap,
            phaseCorrelationMap;
    public int w, h, wh, bW, bH, patchSize, bRW, bRH, sizeWithoutBorders, speedUp, useGAT;
    public float EPSILON;

    public GlobalRedundancy(float[] refPixels, int w, int h, int bW, int bH, float EPSILON, CLContext context,
                            CLCommandQueue queue, int speedUp, int useGAT){
        this.refPixels = refPixels;
        this.w = w;
        this.h = h;
        wh = w * h;
        this.bW = bW;
        this.bH = bH;
        patchSize = bW * bH;
        bRW = bW/2;
        bRH = bH/2;
        sizeWithoutBorders = (w - bRW * 2) * (h - bRH * 2);
        this.EPSILON = EPSILON;
        this.context = context;
        this.queue = queue;
        this.speedUp = speedUp;
        this.useGAT = useGAT;
        localMeans = new float[wh];
        localStds = new float[wh];
        pearsonMap = new float[wh];
        nrmseMap = new float[wh];
        maeMap = new float[wh];
        psnrMap = new float[wh];
        ssimMap = new float[wh];
        huMap = new float[wh];
        entropyMap = new float[wh];
        phaseCorrelationMap = new float[wh];
    }

    public void run(){

        IJ.log("Calculating redundancy...please wait...");

        // ---- Stabilize noise variance using the Generalized Anscombe's transform ----
        if(useGAT == 1) {
            // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
            GATMinimizer minimizer = new GATMinimizer(refPixels, w, h, 0, 100, 0);
            minimizer.run();

            // Get gain, sigma and offset from minimizer and transform pixel values
            refPixels = TransformImageByVST_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);
        }

        // ---- Normalize image ----
        float minMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 1, 2);

        // ---- Write image to GPU ----
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        fillBufferWithFloatArray(clRefPixels, refPixels);
        queue.putWriteBuffer(clRefPixels, false);

        // ---- Calculate local means map ----
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

        // Create, fill and write buffers
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalMeans, localMeans);
        queue.putWriteBuffer(clLocalMeans, false);

        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalStds, localStds);
        queue.putWriteBuffer(clLocalStds, false);

        // Create kernel and set kernel arguments
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");

        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);
        kernelGetLocalMeans.setArg(argn++, clLocalStds);

        // Calculate
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

        // Release resources
        kernelGetLocalMeans.release(); // TODO: MAKES SENSE TO RELEASE KERNELS HERE? WILL THEY BE USED AGAIN?
        programGetLocalMeans.release();

        // ---- Get array of unique StdDev values and a set of coordinates for each, and write to GPU
        float[] stdUnique = getUniqueValues(localStds, w, h, bRW, bRH);
        int[] stdUniqueCoords = getUniqueValueCoordinates(stdUnique, localStds, w, h, bRW, bRH);
        int nUnique = stdUnique.length;

        clUniqueStdCoords = context.createIntBuffer(stdUniqueCoords.length, READ_ONLY);
        fillBufferWithIntArray(clUniqueStdCoords, stdUniqueCoords);
        queue.putWriteBuffer(clUniqueStdCoords, false);

        // ---- Calculate weighted mean Pearson's map ----
        // Create OpenCL program
        String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetPearsonMap.cl");
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BW$", "" + bW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BH$", "" + bH);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRW$", "" + bRW);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRH$", "" + bRH);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$EPSILON$", "" + EPSILON);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$NUNIQUE$", "" + nUnique);
        programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$SPEEDUP$", "" + speedUp);
        programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

        // Create, fill and write buffers
        clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clPearsonMap, pearsonMap);
        queue.putWriteBuffer(clPearsonMap, false);

        // Create kernel and set kernel args
        kernelGetPearsonMap = programGetPearsonMap.createCLKernel("kernelGetPearsonMap");

        argn = 0;
        kernelGetPearsonMap.setArg(argn++, clRefPixels);
        kernelGetPearsonMap.setArg(argn++, clLocalMeans);
        kernelGetPearsonMap.setArg(argn++, clLocalStds);
        kernelGetPearsonMap.setArg(argn++, clUniqueStdCoords);
        kernelGetPearsonMap.setArg(argn++, clPearsonMap);

        // Calculate
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating Pearson's correlations... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetPearsonMap, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }
        // Read the Pearson's map back from the GPU (and finish the mean calculation simultaneously)
        queue.putReadBuffer(clPearsonMap, true);
        for (int y = 0; y<h; y++) {
            for(int x=0; x<w; x++) {
                pearsonMap[y*w+x] = clPearsonMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
                queue.finish();
            }
        }

        // Release resources
        kernelGetPearsonMap.release();
        programGetPearsonMap.release();
        clPearsonMap.release();

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

        // Create kernel and set kernel args
        kernelGetSsimMap = programGetSsimMap.createCLKernel("kernelGetSsimMap");

        argn = 0;
        kernelGetSsimMap.setArg(argn++, clRefPixels);
        kernelGetSsimMap.setArg(argn++, clLocalMeans);
        kernelGetSsimMap.setArg(argn++, clLocalStds);
        kernelGetSsimMap.setArg(argn++, clUniqueStdCoords);
        kernelGetSsimMap.setArg(argn++, clSsimMap);

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

        // Release resources
        kernelGetSsimMap.release();
        programGetSsimMap.release();
        clSsimMap.release();

        // Remap values of non-unique pixels to the corresponding redundancy value
        if(speedUp == 1) {
            ssimMap = remapPixels(ssimMap, w, h, localStds, stdUnique, stdUniqueCoords, nUnique, bRW, bRH);
        }

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
*/

        IJ.log("Done!");
        IJ.log("--------");

        // ---- Display results ----
        IJ.log("Preparing results for display...");

        // Pearson's map (normalized to [0,1])
        float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
        float[] pearsonMapNorm = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);
        FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMapNorm);

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

        // SSIM map (normalized to [0,1])
        float[] ssimMinMax = findMinMax(ssimMap, w, h, bRW, bRH);
        float[] ssimMapNorm = normalize(ssimMap, w, h, bRW, bRH, ssimMinMax, 0, 0);
        FloatProcessor fp5 = new FloatProcessor(w, h, ssimMapNorm);

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

        // Create image stack holding the redundancy maps and display it
        ImageStack ims = new ImageStack(w, h);
        FloatProcessor inputImage = new FloatProcessor(w, h, refPixels);
        ims.addSlice("Variance-stabilised image", inputImage);
        ims.addSlice("Pearson", fp1);
        ims.addSlice("NRMSE", fp2);
        ims.addSlice("MAE", fp3);
        ims.addSlice("PSNR", fp4);
        ims.addSlice("SSIM", fp5);
        ims.addSlice("Hu", fp6);
        ims.addSlice("Entropy", fp7);
        FloatProcessor fp9 = new FloatProcessor(w, h, localStds);
        ims.addSlice("Local stds", fp9);

        ImagePlus ip0 = new ImagePlus("Redundancy Maps", ims);
        ip0.show();
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
}
