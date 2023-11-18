import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.UserFunction;
import ij.process.FloatProcessor;
import ij.process.IntProcessor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Float.NaN;
import static java.lang.Math.*;

public class GlobalRedundancy implements Runnable, UserFunction{

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;
    static private CLCommandQueue queue;
    static private CLProgram programGetLocalMeans, programGetPearsonMap, programGetDiffStdMap;
    static private CLKernel kernelGetLocalMeans, kernelGetPearsonMap, kernelGetDiffStdMap;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clDiffStdMap, clWeightsSumMap;


    // -------------------------- //
    // ---- Image parameters ---- //
    // -------------------------- //

    public float[] refPixels, localMeans, localStds, weightsSumMap;
    public int w, h, wh, bW, bH, patchSize, bRW, bRH, sizeWithoutBorders, scaleFactor, level, w0, h0;
    public float filterConstant, EPSILON;
    public String metric;

    public GlobalRedundancy(float[] refPixels, int w, int h, int bW, int bH, int bRW, int bRH, int patchSize, float EPSILON, CLContext context,
                            CLCommandQueue queue, int scaleFactor, float filterConstant, int level, int w0, int h0, String metric){

        this.refPixels = refPixels;
        this.w = w;
        this.h = h;
        wh = w * h;
        this.bW = bW;
        this.bH = bH;
        this.bRW = bRW;
        this.bRH = bRH;
        sizeWithoutBorders = (w-bRW*2)*(h-bRH*2);
        this.patchSize = patchSize; // Number of pixels in an elliptical patch
        this.EPSILON = EPSILON;
        this.context = context;
        this.queue = queue;
        this.scaleFactor = scaleFactor;
        this.filterConstant = filterConstant;
        this.level = level;
        this.w0 = w0;
        this.h0 = h0;
        localMeans = new float[wh];
        localStds = new float[wh];
        weightsSumMap = new float[wh];
        this.metric = metric;
    }

    @Override
    public void run(){

        IJ.log("Calculating at scale level "+level+"...");


        // ------------------------------------------------------------------------------------------------------ //
        // ---- Write input image (variance-stabilized, mean-subtracted and normalized) to the OpenCL device ---- //
        // ------------------------------------------------------------------------------------------------------ //

        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        fillBufferWithFloatArray(clRefPixels, refPixels);
        queue.putWriteBuffer(clRefPixels, true);


        // ------------------------------------------------------------------ //
        // ---- Calculate local means and local standard deviations maps ---- //
        // ------------------------------------------------------------------ //

        // Create OpenCL program
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRW$", "" + bRW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRH$", "" + bRH);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$EPSILON$", "" + EPSILON);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();

        // Create, fill and write OpenCL buffers
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalMeans, localMeans);
        queue.putWriteBuffer(clLocalMeans, true);

        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalStds, localStds);
        queue.putWriteBuffer(clLocalStds, true);

        // Create OpenCL kernel and set kernel arguments
        kernelGetLocalMeans = programGetLocalMeans.createCLKernel("kernelGetLocalMeans");

        int argn = 0;
        kernelGetLocalMeans.setArg(argn++, clRefPixels);
        kernelGetLocalMeans.setArg(argn++, clLocalMeans);
        kernelGetLocalMeans.setArg(argn++, clLocalStds);

        // Calculate local means and standard deviations map in the GPU
        showStatus("Calculating local means...");
        queue.put2DRangeKernel(kernelGetLocalMeans, 0, 0, w, h, 0, 0);
        queue.finish();
/*
        // Read the local means map back from the GPU
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                localMeans[y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
                queue.finish();
            }
        }
*/
        // Calculate the local variances map
        queue.putReadBuffer(clLocalStds, true);

        float[] localVars = new float[wh];
        for (int y=bRH; y<h-bRH; y++) {
            for (int x=bRW; x<w-bRW; x++) {
                float std = clLocalStds.getBuffer().get(y*w+x);
                localVars[y*w+x] = std*std;
                queue.finish();
            }
        }

        // Release resources
        kernelGetLocalMeans.release();
        programGetLocalMeans.release();


        // ------------------------------------------ //
        // ---- Calculate average noise variance ---- //
        // ------------------------------------------ //

        // Copy the local variances map without borders
        float[] sortedVars = new float[sizeWithoutBorders];
        int index = 0;
        for(int j=bRH; j<h-bRH; j++){
            for(int i=bRW; i<w-bRW; i++){
                sortedVars[index] = localVars[j*w+i];
                index++;
            }
        }

        // Get the 3% lowest variances and calculate their average
        Arrays.sort(sortedVars);
        int nVars = (int) (0.03f * (float)sizeWithoutBorders + 1.0f); // Number of blocks corresponding to the 3% chosen
        float noiseVar = 0.0f;
        for(int i=0; i<nVars; i++){
            noiseVar += sortedVars[i];
        }
        noiseVar = abs(noiseVar/(float)nVars);


        // ---------------------------------------------------------------------- //
        // ---- Calculate noise variance filter mask (a.k.a. relevance mask) ---- //
        // ---------------------------------------------------------------------- //

        double nPixels = 0.0; // Number of relevant pixels
        float threshold = noiseVar*filterConstant;
        int[] relevanceMap = new int[wh];

        for(int j=bRH; j<h-bRH; j++){
            for(int i=bRW; i<w-bRW; i++){
                if(localVars[j*w+i]<threshold){
                    relevanceMap[j*w+i] = 0;
                }else{
                    relevanceMap[j*w+i] = 1;
                    nPixels += 1.0;
                }
            }
        }

        // ----------------------------------------- //
        // ---- Calculate Global Repetition Map ---- //
        // ----------------------------------------- //

        float[] repetitionMap = new float[wh];

        if(metric == "Pearson's R"){

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
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTERCONSTANT$", "" + filterConstant);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$EPSILON$", "" + EPSILON);
            programGetPearsonMap = context.createProgram(programStringGetPearsonMap).build();

            // Create, fill and write OpenCL buffers
            clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clPearsonMap, repetitionMap);
            queue.putWriteBuffer(clPearsonMap, true);

            clWeightsSumMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clWeightsSumMap, weightsSumMap);
            queue.putWriteBuffer(clWeightsSumMap, true);

            // Create OpenCL kernel and set kernel args
            kernelGetPearsonMap = programGetPearsonMap.createCLKernel("kernelGetPearsonMap");

            argn = 0;
            kernelGetPearsonMap.setArg(argn++, clRefPixels);
            kernelGetPearsonMap.setArg(argn++, clLocalMeans);
            kernelGetPearsonMap.setArg(argn++, clLocalStds);
            kernelGetPearsonMap.setArg(argn++, clWeightsSumMap);
            kernelGetPearsonMap.setArg(argn++, clPearsonMap);

            // Calculate weighted mean Pearson's map
            int nXBlocks = w/64 + ((w%64==0)?0:1);
            int nYBlocks = h/64 + ((h%64==0)?0:1);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, h - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, w - nXB * 64);
                    showStatus("Calculating redundancy... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                    queue.put2DRangeKernel(kernelGetPearsonMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                    queue.finish();
                }
            }

            // Read the weighted mean Pearson's map back from the OpenCL device (and finish the mean calculation simultaneously)
            queue.putReadBuffer(clPearsonMap, true);
            queue.putReadBuffer(clWeightsSumMap, true);
            for (int y=bRH; y<h-bRH; y++) {
                for (int x=bRW; x<w-bRW; x++) {
                    repetitionMap[y*w+x] = clPearsonMap.getBuffer().get(y*w+x) / (clWeightsSumMap.getBuffer().get(y*w+x)*(float)nPixels+EPSILON);
                    queue.finish();
                }
            }

            // Release GPU resources
            kernelGetPearsonMap.release();
            programGetPearsonMap.release();
            clPearsonMap.release();
            clWeightsSumMap.release();


            // -------------------------- //
            // ---- Normalize output ---- //
            // -------------------------- //

            // Find min and max
            float min_intensity = Float.MAX_VALUE;
            float max_intensity = -Float.MAX_VALUE;
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1) {
                        float pixelValue = repetitionMap[j*w+i];
                        max_intensity = max(max_intensity, pixelValue);
                        min_intensity = min(min_intensity, pixelValue);
                    }
                }
            }

            // Remap pixels
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1) {
                        repetitionMap[j*w+i] = (repetitionMap[j*w+i]-min_intensity)/(max_intensity-min_intensity+EPSILON);
                    }
                }
            }
        }

        if(metric == "Abs. Diff. of StdDevs"){

            // Create OpenCL program
            String programStringGetDiffStdMap = getResourceAsString(RedundancyMap_.class, "kernelGetDiffStdMap.cl");
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$WIDTH$", "" + w);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$HEIGHT$", "" + h);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRW$", "" + bRW);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRH$", "" + bRH);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$FILTERPARAM$", "" + noiseVar);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$FILTERCONSTANT$", "" + filterConstant);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$EPSILON$", "" + EPSILON);
            programGetDiffStdMap = context.createProgram(programStringGetDiffStdMap).build();

            // Create, fill and write OpenCL buffers
            clDiffStdMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clDiffStdMap, repetitionMap);
            queue.putWriteBuffer(clDiffStdMap, true);

            clWeightsSumMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clWeightsSumMap, weightsSumMap);
            queue.putWriteBuffer(clWeightsSumMap, true);

            // Create OpenCL kernel and set kernel args
            kernelGetDiffStdMap = programGetDiffStdMap.createCLKernel("kernelGetDiffStdMap");

            argn = 0;
            kernelGetDiffStdMap.setArg(argn++, clRefPixels);
            kernelGetDiffStdMap.setArg(argn++, clLocalStds);
            kernelGetDiffStdMap.setArg(argn++, clWeightsSumMap);
            kernelGetDiffStdMap.setArg(argn++, clDiffStdMap);

            // Calculate weighted mean Pearson's map
            int nXBlocks = w/64 + ((w%64==0)?0:1);
            int nYBlocks = h/64 + ((h%64==0)?0:1);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, h - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, w - nXB * 64);
                    showStatus("Calculating redundancy... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                    queue.put2DRangeKernel(kernelGetDiffStdMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                    queue.finish();
                }
            }
            queue.finish();

            // Read the weighted mean Pearson's map back from the OpenCL device (and finish the mean calculation simultaneously)
            queue.putReadBuffer(clDiffStdMap, true);
            queue.putReadBuffer(clWeightsSumMap, true);
            for (int y=bRH; y<h-bRH; y++) {
                for (int x=bRW; x<w-bRW; x++) {

                    // Read similarity value from OpenCL device
                    float similarity = clDiffStdMap.getBuffer().get(y*w+x);
                    queue.finish();

                    // Read weight sum from OpenCL device
                    float weightSum = clWeightsSumMap.getBuffer().get(y*w+x);
                    queue.finish();

                    repetitionMap[y*w+x] = similarity / (weightSum*(float)nPixels+EPSILON);
                }
            }

            // Release GPU resources
            kernelGetDiffStdMap.release();
            programGetDiffStdMap.release();
            clDiffStdMap.release();
            clWeightsSumMap.release();


            // ------------------------------------------- //
            // ---- Normalize output (robust scaling) ---- //
            // ------------------------------------------- //

            // Find median and interquartile range (IQR) for the relevant pixels
            List<Float> relevantValues = new ArrayList<>();

            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    if(relevanceMap[j*w+i]==1){
                        relevantValues.add(repetitionMap[j*w+i]);
                    }
                }
            }

            float[] relevantArray = new float[relevantValues.size()];
            for(int k=0; k<relevantValues.size(); k++){
                relevantArray[k] = relevantValues.get(k);
            }

            float median = calculateMedian(relevantArray);
            float q1 = calculatePercentile(relevantArray, 25);
            float q3 = calculatePercentile(relevantArray, 75);
            float iqr = q3-q1;

            // Robust scaling for the relevant subset
            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    if(relevanceMap[j*w+i]==1){
                        repetitionMap[j*w+i] = (repetitionMap[j*w+i]-median)/(iqr+EPSILON);
                    }else{
                        repetitionMap[j*w+i] = 0.0f;
                    }
                }
            }



            /*
            // Find min and max
            float min_intensity = Float.MAX_VALUE;
            float max_intensity = -Float.MAX_VALUE;
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1) {
                        float pixelValue = repetitionMap[j*w+i];
                        max_intensity = max(max_intensity, pixelValue);
                        min_intensity = min(min_intensity, pixelValue);
                    }
                }
            }

            // Remap pixels
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1) {
                        repetitionMap[j*w+i] = (repetitionMap[j*w+i]-min_intensity)/(max_intensity-min_intensity+EPSILON);
                    }
                }
            }


            */
        }


        // ------------------------- //
        // ---- Display results ---- //
        // ------------------------- //

        IJ.log("Preparing results for display...");

        FloatProcessor fp1 = new FloatProcessor(w, h, repetitionMap);
        ImagePlus imp1 = new ImagePlus("Redundancy Map", fp1);
        imp1.show();

        FloatProcessor impp = new FloatProcessor(w, h, relevanceMap);
        ImagePlus imp2 = new ImagePlus("Relevance Map", impp);
        imp2.show();


/*
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

        ImagePlus impFinal = new ImagePlus("Redundancy Maps (level = " + level + ")", imsFinal);
        impFinal.show();

        // Apply LUT
        IJ.run(impFinal, "mpl-inferno", "");
        */
    }

    @Override
    public double userFunction(double[] doubles, double v) {
        return 0;
    }

    // ---- HELPER FUNCTIONS ----

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

    public static String replaceFirst(String source, String target, String replacement) {
        int index = source.indexOf(target);
        if (index == -1) {
            return source;
        }

        return source.substring(0, index)
                .concat(replacement)
                .concat(source.substring(index+target.length()));
    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

    public static float calculateMedian(float[] values){
        Arrays.sort(values);
        int middle = values.length/2;
        return values.length % 2 == 0 ? (values[middle-1]+values[middle])/2 : values[middle];
    }

    public static float calculatePercentile(float[] values, int percentile){
        int index = (int) Math.ceil((percentile/100.0)*values.length);
        return values[index-1];
    }
}
