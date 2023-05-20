import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.UserFunction;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
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
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetDiffStdMap, programGetWeightsSumMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetDiffStdMap, kernelGetWeightsSumMap;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clDiffStdMap, clWeightsSumMap;

    private CLBuffer<IntBuffer> clUniqueStdCoords;


    // -------------------------- //
    // ---- Image parameters ---- //
    // -------------------------- //

    public float[] refPixels, localMeans, localStds, pearsonMap, diffStdMap, nrmseMap, maeMap, psnrMap, ssimMap, luminanceMap,
            contrastMap, structureMap, huMap, entropyMap, phaseCorrelationMap, hausdorffMap;
    public int w, h, wh, bW, bH, patchSize, bRW, bRH, sizeWithoutBorders, speedUp, useGAT, rotInv, scaleFactor, level, w0, h0;
    public float filterConstant, EPSILON;

    public GlobalRedundancy(float[] refPixels, int w, int h, int bW, int bH, float EPSILON, CLContext context,
                            CLCommandQueue queue, int speedUp, int useGAT, int rotInv, int scaleFactor, float filterConstant, int level, int w0, int h0){
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
        this.filterConstant = filterConstant;
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

        IJ.log("Calculating at scale level "+level+"...");

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
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 0, 0);


        // ------------------------------------------------------------------------- //
        // ---- Estimate noise standard deviation (used for weight calculation) ---- //
        // ------------------------------------------------------------------------- //

        float noiseVar = estimateNoiseVar(refPixels, w, h) + EPSILON;
        System.out.println(noiseVar);
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
        kernelGetLocalMeans.release();
        programGetLocalMeans.release();


        // ----------------------------------------------------------------------------------------------- //
        // ---- Get array of unique StdDev values and a set of coordinates for each, and write to GPU ---- //
        // ----------------------------------------------------------------------------------------------- //
        //TODO:REMOVE THIS
        float[] stdUnique = getUniqueValues(localStds, w, h, bRW, bRH);
        int[] stdUniqueCoords = getUniqueValueCoordinates(stdUnique, localStds, w, h, bRW, bRH);
        int nUnique = stdUnique.length;

        clUniqueStdCoords = context.createIntBuffer(stdUniqueCoords.length, READ_ONLY);
        fillBufferWithIntArray(clUniqueStdCoords, stdUniqueCoords);
        queue.putWriteBuffer(clUniqueStdCoords, false);


        // ----------------------------------- //
        // ---- Calculate weights sum map ---- //
        // ----------------------------------- //

        // Create OpenCL program
        String programStringGetWeightsSumMap = getResourceAsString(RedundancyMap_.class, "kernelGetWeightsSumMap.cl");
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$WIDTH$", "" + w);
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$HEIGHT$", "" + h);
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$BRW$", "" + bRW);
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$BRH$", "" + bRH);
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$FILTERPARAM$", "" + noiseVar);
        programStringGetWeightsSumMap = replaceFirst(programStringGetWeightsSumMap, "$EPSILON$", "" + EPSILON);
        programGetWeightsSumMap = context.createProgram(programStringGetWeightsSumMap).build();

        // Create, fill and write OpenCL buffers
        float[] weightsSumMap = new float[wh];
        clWeightsSumMap = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clWeightsSumMap, weightsSumMap);
        queue.putWriteBuffer(clWeightsSumMap, true);

        // Create OpenCL kernel and set kernel args
        kernelGetWeightsSumMap = programGetWeightsSumMap.createCLKernel("kernelGetWeightsSumMap");

        argn = 0;
        kernelGetWeightsSumMap.setArg(argn++, clLocalStds);
        kernelGetWeightsSumMap.setArg(argn++, clWeightsSumMap);

        // Calculate weights sum map
        for (int nYB = 0; nYB < nYBlocks; nYB++) {
            int yWorkSize = min(64, h - nYB * 64);
            for (int nXB = 0; nXB < nXBlocks; nXB++) {
                int xWorkSize = min(64, w - nXB * 64);
                showStatus("Calculating weights... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                queue.put2DRangeKernel(kernelGetWeightsSumMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the weights sum map back from the device
        queue.putReadBuffer(clWeightsSumMap, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                weightsSumMap[y*w+x] = clWeightsSumMap.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release GPU resources
        kernelGetWeightsSumMap.release();
        programGetWeightsSumMap.release();
        //clDiffStdMap.release();

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
            kernelGetDiffStdMap.setArg(argn++, clWeightsSumMap);
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
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    diffStdMap[y*w+x] = clDiffStdMap.getBuffer().get(y*w+x) / sizeWithoutBorders;
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
/*
            // Filter out regions with low noise variance
            float[] localVars = new float[wh];
            float noiseMeanVar = 0.0f;
            int counter = 0;
            float value;
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
                    if(localVars[j*w+i]<noiseMeanVar*filterConstant){
                        diffStdMap[j*w+i] = 0.0f;
                        System.out.println("filtered");
                    }
                }
            }
*/
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
            kernelGetPearsonMap.setArg(argn++, clWeightsSumMap);
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
/*
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
                    if(localVars[j*w+i]<noiseMeanVar*filterConstant){
                        pearsonMap[j*w+i] = 0.0f;
                    }
                }
            }

 */
        }


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

        // weights sum
        //float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
        //float[] pearsonMapNorm = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);
        //FloatProcessor fp3 = new FloatProcessor(w, h, weightsSumMap);

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

            // Normalize while avoiding borders (TODO: OVERIMPOSE BORDERS IN ALL SCALES, BASICALLY OVERIMPOSE BORDER OF LAST LEVEL)
            float[] tempImg = (float[]) fpFinal.getPixels();
            float[] tempMinMax = findMinMax(tempImg, w0, h0, newBRW, newBRH);
            tempImg = normalize(tempImg, w0, h0, newBRW, newBRH, tempMinMax, 0, 0);
            fpFinal = new FloatProcessor(w0, h0, tempImg);
            imsFinal.setProcessor(fpFinal, i);
        }

        // Filter out regions with low variance
        float[] upscaledVars = (float[]) imsFinal.getProcessor(3).convertToFloatProcessor().getPixels();
        float noiseMeanVar = 0.0f;
        int counter = 0;
        float value;
        for(int j=0; j<h0; j++){
            for(int i=0; i<w0; i++){
                if(i<newBRW || i>=w0-newBRW || j<newBRH || j>=h0-newBRH){
                    continue;
                }
                value = upscaledVars[j*w0+i]*upscaledVars[j*w0+i];
                upscaledVars[j*w0+i] = value;
                noiseMeanVar += value;
                counter++;
            }
        }
        noiseMeanVar /= counter;

        for(int n=1; n<nSlices; n++) {
            for (int j=0; j<h0; j++) {
                for (int i=0; i<w0; i++) {
                    if (upscaledVars[j*w0+i] < noiseMeanVar*filterConstant) {
                        imsFinal.getProcessor(n).setf(j*w0+i, 0.0f);
                    }
                }
            }
        }


        // Delete unwanted slices
        if(rotInv==1){
            imsFinal.deleteSlice(3);
            imsFinal.deleteSlice(2); // 3 becomes 2 after deleting previous
        }else{
            imsFinal.deleteSlice(3);
            imsFinal.deleteSlice(1);
        }

        ImagePlus impFinal = new ImagePlus("Redundancy Maps (level = " + level + ")", imsFinal);
        impFinal.show();

        // Apply LUT
        IJ.run(impFinal, "mpl-inferno", "");
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
