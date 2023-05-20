import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;


public class ArtificialBlockRedundancy_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetSynthPatchDiffStd, programGetSynthPatchPearson;

    static private CLKernel kernelGetPatchMeans, kernelGetSynthPatchDiffStd, kernelGetSynthPatchPearson;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPatchPixels, clDiffStdMap, clPearsonMap;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Get all open image titles
        int  nImages = getImageCount();
        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for(int i=0; i<nImages; i++){
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Artificial Block Redundancy");
        gd.addCheckbox("Rotation invariant?", false);
        gd.addChoice("Patch:", titles, titles[0]);
        gd.addChoice("Image:", titles, titles[1]);
        gd.addSlider("Filter constant: ", 0.0f, 5.0f, 1.0f, 0.1f);
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameters
        int rotInv = 0; // Rotation invariant analysis?
        if(gd.getNextBoolean() == true) {
            rotInv = 1;
        }

        String patchTitle = gd.getNextChoice();
        int patchID = 0;
        for(int i=0; i<nImages; i++){
            if(titles[i] == patchTitle){
                patchID = ids[i];
            }
        }

        String imgTitle = gd.getNextChoice();
        int imgID = 0;
        for(int i=0; i<nImages; i++){
            if(titles[i] == imgTitle){
                imgID = ids[i];
            }
        }

        float filterConstant = (float) gd.getNextNumber();


        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        long start = System.currentTimeMillis();


        // ------------------------------------------------- //
        // ---- Get reference patch and some parameters ---- //
        // ------------------------------------------------- //
        ImagePlus imp = WindowManager.getImage(patchID);
        if (imp == null) {
            IJ.error("Patch image not found. Try again.");
            return;
        }
        ImageProcessor ip = imp.getProcessor();
        FloatProcessor fp = ip.convertToFloatProcessor();
        float[] patchPixels = (float[]) fp.getPixels();
        int bW = fp.getWidth(); // Patch width
        int bH = fp.getHeight(); // Patch height
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int patchSize = (2*bRW+1) * (2*bRW+1) - (int) ceil((sqrt(2)*bRW)*(sqrt(2)*bRW)); // Number of pixels in a circular patch

        // Check if patch dimensions are odd
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        ImagePlus imp0 = WindowManager.getImage(imgID);
        if (imp0 == null) {
            IJ.error("Image not found. Try again.");
            return;
        }
        ImageProcessor ip0 = imp0.getProcessor();
        FloatProcessor fp0 = ip0.convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = fp0.getWidth();
        int h = fp0.getHeight();
        int wh = w*h;
        int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        // Patch
        GATMinimizer minimizer = new GATMinimizer(patchPixels, bW, bH, 0, 100, 0);
        minimizer.run();
        patchPixels = TransformImageByVST_.getGAT(patchPixels, minimizer.gain, minimizer.sigma, minimizer.offset);

        // Image
        minimizer = new GATMinimizer(refPixels, w, h, 0, 100, 0);
        minimizer.run();
        refPixels = TransformImageByVST_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);


        // ----------------------------------- //
        // ---- Normalize patch and image ---- //
        // ----------------------------------- //
        float patchMinMax[] = findMinMax(patchPixels, bW, bH, 0, 0);
        patchPixels = normalize(patchPixels, bW, bH, 0, 0, patchMinMax, 0, 1);

        float imgMinMax[] = findMinMax(refPixels, w, h, 0, 0);
        refPixels = normalize(refPixels, w, h, 0, 0, imgMinMax, 0, 1);


        // ----------------------------------- //
        // ---- Get mean-subtracted patch ---- //
        // ----------------------------------- //
        float[] patchMeanVarStd = meanVarStd(patchPixels);
        for(int i=0; i<patchSize; i++){
            patchPixels[i] = patchPixels[i] - patchMeanVarStd[0];
        }


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
        String programStringGetPatchMeans = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetPatchMeans.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixels);

        float[] localMeans = new float[wh];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[wh];
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


        // -------------------------------------------------------------- //
        // ---- Calculate absolute difference of standard deviations ---- //
        // -------------------------------------------------------------- //

        if(rotInv == 1) {
            // Create OpenCL program
            String programStringGetSynthPatchDiffStd = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchDiffStd.cl");
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$WIDTH$", "" + w);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$HEIGHT$", "" + h);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$PATCH_SIZE$", "" + patchSize);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BW$", "" + bW);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BH$", "" + bH);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BRW$", "" + bRW);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BRH$", "" + bRH);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$PATCH_STD$", "" + patchMeanVarStd[2]);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchDiffStd = context.createProgram(programStringGetSynthPatchDiffStd).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(bW*bH, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixels);

            float[] diffStdMap = new float[wh];
            clDiffStdMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clDiffStdMap, diffStdMap);

            // Create OpenCL kernel and set args
            kernelGetSynthPatchDiffStd = programGetSynthPatchDiffStd.createCLKernel("kernelGetSynthPatchDiffStd");

            argn = 0;
            kernelGetSynthPatchDiffStd.setArg(argn++, clPatchPixels);
            kernelGetSynthPatchDiffStd.setArg(argn++, clRefPixels);
            kernelGetSynthPatchDiffStd.setArg(argn++, clLocalMeans);
            kernelGetSynthPatchDiffStd.setArg(argn++, clLocalStds);
            kernelGetSynthPatchDiffStd.setArg(argn++, clDiffStdMap);

            // Calculate
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clDiffStdMap, true);
            queue.put2DRangeKernel(kernelGetSynthPatchDiffStd, 0, 0, w, h, 0, 0);
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

            // Release memory
            kernelGetSynthPatchDiffStd.release();
            clPatchPixels.release();
            clDiffStdMap.release();
            programGetSynthPatchDiffStd.release();


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

            // Display results
            diffStdMinMax = findMinMax(diffStdMap, w, h, bRW, bRH);
            float[] diffStdMapNorm = normalize(diffStdMap, w, h, bRW, bRH, diffStdMinMax, 0, 0);
            FloatProcessor fp1 = new FloatProcessor(w, h, diffStdMapNorm);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }

        if(rotInv == 0) {
            // ----
            // ---- Pearson correlation ----
            String programStringGetSynthPatchPearson = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchPearson.cl");
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$WIDTH$", "" + w);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$HEIGHT$", "" + h);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$PATCH_SIZE$", "" + patchSize);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BW$", "" + bW);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BH$", "" + bH);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BRW$", "" + bRW);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BRH$", "" + bRH);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$PATCH_STD$", "" + patchMeanVarStd[2]);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchPearson = context.createProgram(programStringGetSynthPatchPearson).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(bW*bH, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixels);

            float[] pearsonMap = new float[wh];
            clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clPearsonMap, pearsonMap);

            // Create kernel and set args
            kernelGetSynthPatchPearson = programGetSynthPatchPearson.createCLKernel("kernelGetSynthPatchPearson");

            argn = 0;
            kernelGetSynthPatchPearson.setArg(argn++, clPatchPixels);
            kernelGetSynthPatchPearson.setArg(argn++, clRefPixels);
            kernelGetSynthPatchPearson.setArg(argn++, clLocalMeans);
            kernelGetSynthPatchPearson.setArg(argn++, clLocalStds);
            kernelGetSynthPatchPearson.setArg(argn++, clPearsonMap);

            // Calculate Pearson's correlation coefficient ----
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clPearsonMap, true);
            queue.put2DRangeKernel(kernelGetSynthPatchPearson, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's coefficients back from the GPU
            queue.putReadBuffer(clPearsonMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    pearsonMap[y * w + x] = clPearsonMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetSynthPatchPearson.release();
            clPatchPixels.release();
            clPearsonMap.release();
            programGetSynthPatchPearson.release();

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
        double variance = abs(sq_sum / n - mean * mean); // abs() solves a bug where negative zeros appeared

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

