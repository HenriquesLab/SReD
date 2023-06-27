import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;


public class EmpiricalBlockRedundancy_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetPatchDiffStd, programGetPatchPearson;

    static private CLKernel kernelGetPatchMeans, kernelGetPatchDiffStd, kernelGetPatchPearson;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clDiffStdMap, clPearsonMap;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Empirical Block Redundancy");
        gd.addCheckbox("Rotation invariant?", false);
        gd.addSlider("Filter constant: ", 0.0f, 5.0f, 0.1f);
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameters
        int rotInv = 0; // Rotation invariant analysis?
        if(gd.getNextBoolean() == true) {
            rotInv = 1;
        }

        float filterConstant = (float) gd.getNextNumber();

        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        long start = System.currentTimeMillis();


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }
        ImageProcessor ip0 = imp0.getProcessor();
        FloatProcessor fp0 = ip0.convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
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
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)
        int patchSize = (2*bRW+1) * (2*bRH+1) - (int) ceil((sqrt(2)*bRW)*(sqrt(2)*bRH)); // Number of pixels in a circular patch
        int centerX = bx + bRW; // Reference patch center (x-axis)
        int centerY = by + bRH; // Reference patch center (y-axis)

        // Verify that selected patch dimensions are odd
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
        GATMinimizer minimizer = new GATMinimizer(refPixels, w, h, 0, 100, 0);
        minimizer.run();

        // Get gain, sigma and offset from minimizer and transform pixel values
        refPixels = TransformImageByVST_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);


        // ------------------------------- //
        // ---- Normalize input image ---- //
        // ------------------------------- //

        float minMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, minMax, 0, 1);


        // --------------------------- //
        // ---- Initialize OpenCL ---- //
        // --------------------------- //

        // Check OpenCL devices
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

        // Create OpenCL context
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

        // Create command queue
        queue = chosenDevice.createCommandQueue();
        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        IJ.log("Calculating redundancy...please wait...");


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans = getResourceAsString(EmpiricalBlockRedundancy_.class, "kernelGetPatchMeans.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[w * h];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[w*h];
        fillBufferWithFloatArray(clLocalStds, localStds);

        // Create OpenCL kernel and set args
        kernelGetPatchMeans = programGetPatchMeans.createCLKernel("kernelGetPatchMeans");

        int argn = 0;
        kernelGetPatchMeans.setArg(argn++, clRefPixels);
        kernelGetPatchMeans.setArg(argn++, clLocalMeans);
        kernelGetPatchMeans.setArg(argn++, clLocalStds);

        // Calculate
        queue.putWriteBuffer(clRefPixels, true);
        queue.putWriteBuffer(clLocalMeans, true);
        queue.putWriteBuffer(clLocalStds, true);

        showStatus("Calculating local means...");

        queue.put2DRangeKernel(kernelGetPatchMeans, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the local means map back from the device
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                localMeans[y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
            }
        }
        queue.finish();

        // Read the local stds map back from the device
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


        // ------------------------------ //
        // ---- Calculate redundancy ---- //
        // ------------------------------ //

        if(rotInv == 1) {

            // -------------------------------------------------------------- //
            // ---- Calculate absolute difference of standard deviations ---- //
            // -------------------------------------------------------------- //

            // Create OpenCL program
            String programStringGetPatchDiffStd = getResourceAsString(EmpiricalBlockRedundancy_.class, "kernelGetPatchDiffStd.cl");
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$WIDTH$", "" + w);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$HEIGHT$", "" + h);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$CENTER_X$", "" + centerX);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$CENTER_Y$", "" + centerY);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$BRW$", "" + bRW);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$BRH$", "" + bRH);
            programStringGetPatchDiffStd = replaceFirst(programStringGetPatchDiffStd, "$EPSILON$", "" + EPSILON);
            programGetPatchDiffStd = context.createProgram(programStringGetPatchDiffStd).build();

            // Fill OpenCL buffers
            float[] diffStdMap = new float[wh];
            clDiffStdMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clDiffStdMap, diffStdMap);

            // Create OpenCL kernel and set args
            kernelGetPatchDiffStd = programGetPatchDiffStd.createCLKernel("kernelGetPatchDiffStd");

            argn = 0;
            kernelGetPatchDiffStd.setArg(argn++, clRefPixels);
            kernelGetPatchDiffStd.setArg(argn++, clLocalMeans);
            kernelGetPatchDiffStd.setArg(argn++, clLocalStds);
            kernelGetPatchDiffStd.setArg(argn++, clDiffStdMap);

            // Calculate
            queue.putWriteBuffer(clDiffStdMap, true);
            queue.put2DRangeKernel(kernelGetPatchDiffStd, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read results back from the device
            queue.putReadBuffer(clDiffStdMap, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    diffStdMap[y*w+x] = clDiffStdMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();

            // Invert values (because so far we have inverse frequencies)
            float[] diffStdMinMax = findMinMax(diffStdMap, w, h, bRW, bRH);
            diffStdMap = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);
            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    diffStdMap[j*w+i] = 1.0f - diffStdMap[j*w+i];
                }
            }
            diffStdMap = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);

            // Filter out flat regions
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
                    }
                }
            }

            // Release memory
            kernelGetPatchDiffStd.release();
            clDiffStdMap.release();
            programGetPatchDiffStd.release();

            // Display results
            diffStdMinMax = findMinMax(diffStdMap, w, h, bRW, bRH);
            float[] diffStdMapNorm = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);
            FloatProcessor fp1 = new FloatProcessor(w, h, diffStdMapNorm);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }

        if(rotInv == 0) {

            // ------------------------------------------ //
            // ---- Calculate Pearson's correlations ---- //
            // ------------------------------------------ //

            // Create OpenCL program
            String programStringGetPatchPearson = getResourceAsString(EmpiricalBlockRedundancy_.class, "kernelGetPatchPearson.cl");
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$WIDTH$", "" + w);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$HEIGHT$", "" + h);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_X$", "" + centerX);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$CENTER_Y$", "" + centerY);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRW$", "" + bRW);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRH$", "" + bRH);
            programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$EPSILON$", "" + EPSILON);
            programGetPatchPearson = context.createProgram(programStringGetPatchPearson).build();

            // Fill OpenCL buffers
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

            // Calculate Pearson's coefficients
            queue.putWriteBuffer(clPearsonMap, true);
            queue.put2DRangeKernel(kernelGetPatchPearson, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's coefficients back from the device
            queue.putReadBuffer(clPearsonMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    pearsonMap[y * w + x] = clPearsonMap.getBuffer().get(y * w + x)*clPearsonMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release OpenCL resources
            kernelGetPatchPearson.release();
            clPearsonMap.release();
            programGetPatchPearson.release();

            // Filter out flat regions
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
                        pearsonMap[j*w+i] = 0.0f;
                    }
                }
            }

            // Display results
            float[] pearsonMinMax = findMinMax(pearsonMap, w, h, bRW, bRH);
            float[] pearsonMapNorm = normalize(pearsonMap, w, h, bRW, bRH, pearsonMinMax, 0, 0);
            FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMapNorm);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }


        // -------------------- //
        // ---- Stop timer ---- //
        // -------------------- //

        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }

    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //

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
        double variance = abs(sq_sum / n - mean * mean); // abs() solves a bug where negative zeros appeared

        return new float[]{(float) mean, (float) variance, (float) sqrt(variance)};

    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
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

