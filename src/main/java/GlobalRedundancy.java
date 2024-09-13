/**
 * Calculates the Global Repetition Map. Each local neighbourhood is used as a reference for a round of Block Repetition.
 * Each pairwise comparison is weighted based on the similarity between reference and test blocks, and the resulting Block Repetition Map is averaged.
 * The average value is plotted at the center position of the reference neighbourhood.
 *
 * @author Afonso Mendes
 *
 **/


import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.UserFunction;
import ij.plugin.LutLoader;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.LUT;

import java.awt.image.IndexColorModel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
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

    public float[] refPixels;
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

        float[] localMeans = new float[wh];
        Arrays.fill(localMeans, 0.0f);

        float[] localStds = new float[wh];
        Arrays.fill(localStds, 0.0f);

        // Create OpenCL program
        String programStringGetLocalMeans = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$WIDTH$", "" + w);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$HEIGHT$", "" + h);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRW$", "" + bRW);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$BRH$", "" + bRH);
        programStringGetLocalMeans = replaceFirst(programStringGetLocalMeans, "$EPSILON$", "" + EPSILON);
        programGetLocalMeans = context.createProgram(programStringGetLocalMeans).build();
        System.out.println(programGetLocalMeans.getBuildLog());
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

        // Read the local standard deviations map from the OpenCL device
        queue.putReadBuffer(clLocalStds, true);
        for (int y=bRH; y<h-bRH; y++) {
            for (int x=bRW; x<w-bRW; x++) {
                localStds[y*w+x] = clLocalStds.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release resources
        kernelGetLocalMeans.release();
        programGetLocalMeans.release();


        // --------------------------------- //
        // ---- Calculate Relevance Map ---- //
        // --------------------------------- //

        // Define block dimensions for variance calculation
        int blockWidth, blockHeight;
        int CIF = 352*288; // Resolution of a CIF file

        if(wh<=CIF){
            blockWidth = 8;
            blockHeight = 8;
        }else{
            blockWidth = 16;
            blockHeight = 16;
        }

        int nBlocksX = w / blockWidth; // number of blocks in each row
        int nBlocksY = h / blockHeight; // number of blocks in each column
        int nBlocks = nBlocksX * nBlocksY; // total number of blocks

        // Calculate local variances
        float[] localVars = new float[nBlocks];
        Arrays.fill(localVars, 0.0f);

        int index = 0;

        for(int y=0; y<nBlocksY; y++){
            for(int x=0; x<nBlocksX; x++){
                double[] meanVar = getMeanAndVarBlock(refPixels, w, x*blockWidth, y*blockHeight, (x+1)*blockWidth, (y+1)*blockHeight);
                localVars[index] = (float)meanVar[1];
                index++;
            }
        }

        // Sort the local variances
        float[] sortedVars = new float[nBlocks];
        Arrays.fill(sortedVars, 0.0f);

        index = 0;
        for(int i=0; i<nBlocks; i++){
            sortedVars[index] = localVars[index];
            index++;
        }
        Arrays.sort(sortedVars);

        // Get the 3% lowest variances and calculate their average
        int nVars = (int) (0.03f * (float)nBlocks + 1.0f); // Number of blocks corresponding to 3% of the total amount of blocks
        float noiseVar = 0.0f;

        for(int i=0; i<nVars; i++){
            noiseVar += sortedVars[i];
        }
        noiseVar = abs(noiseVar/(float)nVars);
        noiseVar = (1.0f+0.001f*(noiseVar-40.0f)) * noiseVar;

        // Build the relevance map
        float[] relevanceMap = new float[wh];
        Arrays.fill(relevanceMap, 0.0f);

        float threshold;
        if(noiseVar == 0.0f){
            IJ.log("WARNING: Noise variance is 0. Adjust the relevance threshold using the filter constant directly.");
            threshold = filterConstant;
            IJ.log("Threshold: " + filterConstant);
        }else{
            IJ.log("Noise variance: " + noiseVar);
            threshold = noiseVar*filterConstant;
            IJ.log("Relevance threshold: " + threshold);
        }

        double nPixels = 0.0; // Number of relevant pixels
        for(int j=bRH; j<h-bRH; j++){
            for(int i=bRW; i<w-bRW; i++){
                float var = localStds[j*w+i]*localStds[j*w+i];
                if(var<threshold || var==0.0f){
                    relevanceMap[j*w+i] = 0.0f;
                }else{
                    relevanceMap[j*w+i] = 1.0f;
                    nPixels += 1.0;
                }
            }
        }


        // ----------------------------------------- //
        // ---- Calculate Global Repetition Map ---- //
        // ----------------------------------------- //

        float[] repetitionMap = new float[wh];
        Arrays.fill(repetitionMap, 0.0f);

        float[] weightsSumMap = new float[wh];
        Arrays.fill(weightsSumMap, 0.0f);


        if(metric == "Pearson's R"){

            // Create OpenCL program
            String programStringGetPearsonMap = getResourceAsString(RedundancyMap_.class, "kernelGetGlobalPearson.cl");
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$WIDTH$", "" + w);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$HEIGHT$", "" + h);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRW$", "" + bRW);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$BRH$", "" + bRH);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$FILTERPARAM$", "" + noiseVar);
            programStringGetPearsonMap = replaceFirst(programStringGetPearsonMap, "$THRESHOLD$", "" + threshold);
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

            showStatus("Processing results...");

            // Read the weighted mean Pearson's map back from the OpenCL device (and finish the mean calculation simultaneously)
            queue.putReadBuffer(clPearsonMap, true);
            queue.putReadBuffer(clWeightsSumMap, true);

            for (int y=bRH; y<h-bRH; y++) {
                for (int x=bRW; x<w-bRW; x++) {

                    // Read the similarity value from the OpenCL device
                    float similarity = clPearsonMap.getBuffer().get(y*w+x);
                    queue.finish();

                    // Read the weight sum from the OpenCL device
                    float weightSum = clWeightsSumMap.getBuffer().get(y*w+x);
                    queue.finish();

                    // Calculate and store the repetition score
                    repetitionMap[y*w+x] = similarity / (weightSum * (float)nPixels + EPSILON);
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

            showStatus("Normalizing output...");

            // Find min and max
            float min_intensity = Float.MAX_VALUE;
            float max_intensity = -Float.MAX_VALUE;
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i]==1.0f) {
                        float pixelValue = repetitionMap[j*w+i];
                        max_intensity = max(max_intensity, pixelValue);
                        min_intensity = min(min_intensity, pixelValue);
                    }
                }
            }

            // Remap pixels
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i]==1.0f) {
                        repetitionMap[j*w+i] = (repetitionMap[j*w+i]-min_intensity)/(max_intensity-min_intensity+EPSILON);
                    }else{
                        repetitionMap[j*w+i] = 0.0f;
                    }
                }
            }
        }

        if(metric == "Cosine similarity"){

            // Create OpenCL program
            String programStringGetDiffStdMap = getResourceAsString(RedundancyMap_.class, "kernelGetDiffStdMap.cl");
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$WIDTH$", "" + w);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$HEIGHT$", "" + h);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRW$", "" + bRW);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$BRH$", "" + bRH);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$FILTERPARAM$", "" + noiseVar);
            programStringGetDiffStdMap = replaceFirst(programStringGetDiffStdMap, "$THRESHOLD$", "" + threshold);
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

            // Calculate weighted repetition map
            int nXBlocks = w/64 + ((w%64==0)?0:1);
            int nYBlocks = h/64 + ((h%64==0)?0:1);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, h - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, w - nXB * 64);
                    showStatus("Calculating global repetition... blockX=" + nXB + "/" + nXBlocks + " blockY=" + nYB + "/" + nYBlocks);
                    queue.put2DRangeKernel(kernelGetDiffStdMap, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                    queue.finish();
                }
            }
            queue.finish();

            showStatus("Retrieving results...");

            // Read the weighted repetition map back from the OpenCL device (and finish the mean calculation simultaneously)
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

                    // Store the repetition score
                    repetitionMap[y*w+x] = similarity / (weightSum*(float)nPixels+EPSILON);

                }
            }

            // Release GPU resources
            kernelGetDiffStdMap.release();
            programGetDiffStdMap.release();
            clDiffStdMap.release();
            clWeightsSumMap.release();


            // -------------------------- //
            // ---- Normalize output ---- //
            // -------------------------- //

            showStatus("Normalizing output...");

            // Find min and max
            float min_intensity = Float.MAX_VALUE;
            float max_intensity = -Float.MAX_VALUE;
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1.0f) {
                        float pixelValue = repetitionMap[j*w+i];
                        max_intensity = max(max_intensity, pixelValue);
                        min_intensity = min(min_intensity, pixelValue);
                    }
                }
            }

            // Remap pixels
            for(int j=bRH; j<h-bRH; j++) {
                for(int i=bRW; i<w-bRW; i++) {
                    if(relevanceMap[j*w+i] == 1.0f) {
                        repetitionMap[j*w+i] = (repetitionMap[j*w+i]-min_intensity)/(max_intensity-min_intensity+EPSILON);
                    }else{
                        repetitionMap[j*w+i] = 0.0f;
                    }
                }
            }
        }


        // Scale back to original dimensions

        int newBRW = bRW * scaleFactor;
        int newBRH = bRH * scaleFactor;

        // Get output without borders (so that we later dont interpolate edges)
        FloatProcessor fpOutput = new FloatProcessor(w, h, repetitionMap);
        fpOutput.setRoi(bRW, bRH, w-2*bRW, h-2*bRH);
        FloatProcessor fpCrop = fpOutput.crop().convertToFloatProcessor(); // If you don't create a new one, bad things happen

        // Upscale
        fpCrop.setInterpolationMethod(ImageProcessor.BILINEAR); // Bicubic is generating edge artifacts
        FloatProcessor fpUpscaled = fpCrop.resize(w0-2*newBRW, h0-2*newBRH, true).convertToFloatProcessor(); // If you don't crete a new one, bad things happen

        // Add borders
        FloatProcessor fpFinal = new FloatProcessor(w0, h0);
        fpFinal.insert(fpUpscaled, newBRW, newBRH);

        // Normalize while avoiding borders
        float[] temp = (float[]) fpFinal.getPixels();
        float min_intensity = Float.MAX_VALUE;
        float max_intensity = -Float.MAX_VALUE;

        for(int y=newBRH; y<h0-newBRH; y++){
            for(int x=newBRW; x<w0-newBRW; x++){
                float pixelValue = temp[y*w0+x];
                max_intensity = max(max_intensity, pixelValue);
                min_intensity = min(min_intensity, pixelValue);
            }
        }

        // Remap pixels
        for(int y=newBRH; y<h0-newBRH; y++){
            for(int x=newBRW; x<w0-newBRW; x++){
                fpFinal.setf(x, y, (temp[y*w0+x]-min_intensity)/(max_intensity-min_intensity+EPSILON));
            }
        }


        // ------------------------- //
        // ---- Display results ---- //
        // ------------------------- //

        IJ.log("Preparing results for display...");

        ImagePlus imp1 = new ImagePlus("Repetition Map - Level "+level, fpFinal);

        // Apply SReD LUT
        InputStream lutStream = getClass().getResourceAsStream("/luts/sred-jet.lut");
        if (lutStream == null) {
            IJ.error("Could not load SReD LUT. Using default LUT.");
        }else{
            try {
                // Load LUT file
                IndexColorModel icm = LutLoader.open(lutStream);
                byte[] r = new byte[256];
                byte[] g = new byte[256];
                byte[] b = new byte[256];
                icm.getReds(r);
                icm.getGreens(g);
                icm.getBlues(b);
                LUT lut = new LUT(8, 256, r, g, b);

                // Apply LUT to image
                imp1.getProcessor().setLut(lut);
                //imp1.updateAndDraw();
            } catch (IOException e) {
                IJ.error("Could not load SReD LUT");
            }
        }

        imp1.show();

/*


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

    // Get mean and variance of a patch
    public double[] getMeanAndVarBlock(float[] pixels, int width, int xStart, int yStart, int xEnd, int yEnd) {
        double mean = 0;
        double var;

        double sq_sum = 0;

        int bWidth = xEnd-xStart;
        int bHeight = yEnd - yStart;
        int bWH = bWidth*bHeight;

        for (int j=yStart; j<yEnd; j++) {
            for (int i=xStart; i<xEnd; i++) {
                float v = pixels[j*width+i];
                mean += v;
                sq_sum += v * v;
            }
        }

        mean = mean / bWH;
        var = sq_sum / bWH - mean * mean;

        return new double[] {mean, var};
    }
}
