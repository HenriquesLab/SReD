import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
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

    static private CLProgram programGetPatchMeans, programGetSynthPatchDiffStd, programGetSynthPatchPearson,
            programGetSynthPatchHu, programGetSynthPatchSsim, programGetRelevanceMap;

    static private CLKernel kernelGetPatchMeans, kernelGetSynthPatchDiffStd, kernelGetSynthPatchPearson,
            kernelGetSynthPatchHu, kernelGetSynthPatchSsim, kernelGetRelevanceMap;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPatchPixels, clDiffStdMap, clPearsonMap,
            clHuMap, clSsimMap, clRelevanceMap, clGaussianWindow;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Get all open image titles
        int  nImages = getImageCount();
        if (nImages < 2) {
            IJ.error("Not enough images found. You need at least two.");
            return;
        }

        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for(int i=0; i<nImages; i++){
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Define metric possibilities
        String[] metrics = new String[4];
        metrics[0] = "Pearson's R";
        metrics[1] = "Abs. Diff. of StdDevs";
        metrics[2] = "Hu moments";
        metrics[3] = "mSSIM";

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Artificial Block Redundancy");
        gd.addChoice("Patch:", titles, titles[0]);
        gd.addChoice("Image:", titles, titles[1]);
        gd.addSlider("Filter constant: ", 0.0f, 5.0f, 1.0f, 0.1f);
        gd.addChoice("Metric:", metrics, metrics[0]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameters from dialog box
        String patchTitle = gd.getNextChoice();
        int patchID = 0;
        for(int i=0; i<nImages; i++){
            if(titles[i].equals(patchTitle)){ // .equals() instead of "==" required to run from macro
                patchID = ids[i];
            }
        }

        String imgTitle = gd.getNextChoice();
        int imgID = 0;
        for(int i=0; i<nImages; i++){
            if(titles[i].equals(imgTitle)){ // .equals() instead of "==" required to run from macro
                imgID = ids[i];
            }
        }

        float filterConstant = (float) gd.getNextNumber();

        String metric = gd.getNextChoice();

        boolean normalizeOutput = gd.getNextBoolean();

        boolean useDevice = gd.getNextBoolean();


        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        IJ.log("SReD has started, please wait.");
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

        // Check if patch dimensions are odd, otherwise kill program
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)


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
        //int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)


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

/*
        // ----------------------------------- //
        // ---- Calculate gaussian window ---- //
        // ----------------------------------- //

        // Define parameters
        double[] gaussianWindow = new double[bW*bH]; // The full window before keeping only the pixels within the inbound circle/ellipse
        double gaussianSum = 0.0;
        double sigma_x = bW/1.0; // Gaussian sigma in the x-direction (SSIM paper uses 7.3 instead of 4)
        double sigma_y = bH/1.0; // Gaussian sigma in the y-direction

        // Calculate gaussian window
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                double x = (double)(i-bRW)/sigma_x;
                double y = (double)(j-bRH)/sigma_y;
                gaussianWindow[j*bW+i] = Math.exp(-(x*x+y*y)/2.0);
                gaussianSum += gaussianWindow[j*bW+i];
            }
        }

        // Normalize window to sum=1
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                gaussianWindow[j*bW+i] /= gaussianSum;
            }
        }
*/

        // --------------------------------- //
        // ---- Process reference patch ---- //
        // --------------------------------- //

        // Get final patch size (after removing pixels outside inbound circle/ellipse)
        int patchSize = 0;
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                float dx = (float)(i-bRW);
                float dy = (float)(j-bRH);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                    patchSize++;
                }
            }
        }

        // Convert patch to "double" type (keeping only the pixels within the inbound circle/ellipse)
        // Also, apply Gaussian Window
        double[] patchPixelsDouble = new double[patchSize];
        //double[] gaussianWindowDouble = new double[patchSize];
        int index = 0;
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                float dx = (float)(i-bRW);
                float dy = (float)(j-bRH);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                    patchPixelsDouble[index] = (double) patchPixels[j*bW+i];
                    //patchPixelsDouble[index] = (double) patchPixels[j*bW+i] * gaussianWindow[j*bW+i];
                    //gaussianWindowDouble[index] = gaussianWindow[j*bW+i];
                    index++;
                }
            }
        }

        // Find min and max
        double patchMin = Double.MAX_VALUE; // Initialize as a very large number
        double patchMax = -Double.MAX_VALUE; // Initialize as a very small number

        for(int i=0; i<patchSize; i++){
            patchMin = min(patchMin, patchPixelsDouble[i]);
            patchMax = max(patchMax, patchPixelsDouble[i]);
        }

        // Normalize and calculate mean
        double patchMean = 0.0;
        for(int i=0; i<patchSize; i++){
            patchPixelsDouble[i] = (patchPixelsDouble[i] - patchMin)/(patchMax - patchMin + EPSILON);
            patchMean += patchPixelsDouble[i];

        }
        patchMean /= (double) patchSize;

        // Subtract mean
        for(int i=0; i<patchSize; i++){
            patchPixelsDouble[i] = patchPixelsDouble[i] - patchMean;
        }

        // Normalize again
        patchMin = Double.MAX_VALUE; // Initialize as a very large number
        patchMax = -Double.MAX_VALUE; // Initialize as a very small number

        for(int i=0; i<patchSize; i++){
            patchMin = min(patchMin, patchPixelsDouble[i]);
            patchMax = max(patchMax, patchPixelsDouble[i]);
        }

        for(int i=0; i<patchSize; i++){
            patchPixelsDouble[i] = (patchPixelsDouble[+i] - patchMin)/(patchMax - patchMin + EPSILON);
        }

        // Typecast back to float
        float[] patchPixelsFloat = new float[patchSize];
        //float[] gaussianWindowFloat = new float[patchSize];
        for(int i=0; i<patchSize; i++){
            patchPixelsFloat[i] = (float)patchPixelsDouble[i];
            //gaussianWindowFloat[i] = (float)gaussianWindowDouble[i];
        }

        // Calculate mean and standard deviation
        float patchMeanFloat = 0.0f;
        double patchStdDev = 0.0;
        for(int i=0; i<patchSize; i++){
            patchMeanFloat += patchPixelsFloat[i];
            patchStdDev += (patchPixelsDouble[i] - patchMean) * (patchPixelsDouble[i] - patchMean);
        }
        patchMeanFloat /= (float) patchSize;
        patchStdDev = (float) sqrt(patchStdDev/(patchSize-1));


        // ----------------------- //
        // ---- Process image ---- //
        // ----------------------- //

        // Cast to "double" type
        double[] refPixelsDouble = new double[wh];
        for(int i=0; i<wh; i++){
            refPixelsDouble[i] = (double)refPixels[i];
        }

        // Get min and max
        double imgMin = Double.MAX_VALUE;
        double imgMax = -Double.MAX_VALUE;
        for(int i=0; i<wh; i++){
            double pixelValue = refPixelsDouble[i];
            if(pixelValue<imgMin){
                imgMin = pixelValue;
            }
            if(pixelValue>imgMax){
                imgMax = pixelValue;
            }
        }

        // Normalize
        for(int i=0; i<wh; i++) {
            refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + EPSILON);
        }

        // Cast back to float
        for(int i=0; i<wh; i++){
            refPixels[i] = (float)refPixelsDouble[i];
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

        // Get chosen device from preferences
        if(useDevice){
            String deviceName = Prefs.get("SReD.OpenCL.device", null);
            for (CLDevice device : allDevices) {
                if (device.getName().equals(deviceName)) {
                    chosenDevice = device;
                    break;
                }
            }
        }

        IJ.log("Chosen device: " + chosenDevice.getName());
        IJ.log("--------");

        // Create command queue
        queue = chosenDevice.createCommandQueue();
        int elementCount = w*h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        IJ.log("Calculating redundancy...");


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        //clGaussianWindow = context.createFloatBuffer(patchSize, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetPatchMeans.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$EPSILON$", "" + EPSILON);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixels);

        //fillBufferWithFloatArray(clGaussianWindow, gaussianWindowFloat);

        float[] localMeans = new float[wh];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[wh];
        fillBufferWithFloatArray(clLocalStds, localStds);

        // Create OpenCL kernel and set args
        kernelGetPatchMeans = programGetPatchMeans.createCLKernel("kernelGetPatchMeans");

        int argn = 0;
        kernelGetPatchMeans.setArg(argn++, clRefPixels);
        //kernelGetPatchMeans.setArg(argn++, clGaussianWindow);
        kernelGetPatchMeans.setArg(argn++, clLocalMeans);
        kernelGetPatchMeans.setArg(argn++, clLocalStds);

        // Calculate
        queue.putWriteBuffer(clRefPixels, true);
        //queue.putWriteBuffer(clGaussianWindow, true);
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


        // --------------------------------------------------------------- //
        // ---- Calculate block repetition map with the chosen metric ---- //
        // --------------------------------------------------------------- //

        if(metric == metrics[0]) { // Pearson correlation
            showStatus("Calculating Pearson correlations...");

            // Build OpenCL program
            String programStringGetSynthPatchPearson = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchPearson.cl");
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$WIDTH$", "" + w);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$HEIGHT$", "" + h);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$PATCH_SIZE$", "" + patchSize);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BW$", "" + bW);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BH$", "" + bH);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BRW$", "" + bRW);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$BRH$", "" + bRH);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$PATCH_MEAN$", "" + patchMeanFloat);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$PATCH_STD$", "" + patchStdDev);
            programStringGetSynthPatchPearson = replaceFirst(programStringGetSynthPatchPearson, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchPearson = context.createProgram(programStringGetSynthPatchPearson).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

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

            // Calculate Pearson's correlation coefficient (reference patch vs. all)
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clPearsonMap, true);
            queue.put2DRangeKernel(kernelGetSynthPatchPearson, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's coefficients back from the GPU
            queue.putReadBuffer(clPearsonMap, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    pearsonMap[y*w+x] = clPearsonMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetSynthPatchPearson.release();
            clPatchPixels.release();
            clPearsonMap.release();
            programGetSynthPatchPearson.release();


            // --------------------------------------- //
            // ---- Filter out irrelevant regions ---- //
            // --------------------------------------- //
            //TODO: SHOULDNT THIS BE 3% PRIME ASIA?
            // NOTE: THIS KERNEL IS THE SAME AS THE LOCAL STDS BUT WITHOUT NORMALIZING THE PATCHES. WE CAN USE THE SAME BUFFERS

            if(filterConstant>0.0f) {
                showStatus("Calculating relevance map...");

                // Create OpenCL program
                String programStringGetRelevanceMap = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetRelevanceMap.cl");
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$WIDTH$", "" + w);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$HEIGHT$", "" + h);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$PATCH_SIZE$", "" + patchSize);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRW$", "" + bRW);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRH$", "" + bRH);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$EPSILON$", "" + EPSILON);
                programGetRelevanceMap = context.createProgram(programStringGetRelevanceMap).build();

                // Create and fill buffers
                float[] relevanceMap = new float[wh];
                clRelevanceMap = context.createFloatBuffer(wh, READ_WRITE);
                fillBufferWithFloatArray(clRelevanceMap, relevanceMap);
                queue.putWriteBuffer(clRelevanceMap, true);
                queue.finish();

                // Create OpenCL kernel and set args
                kernelGetRelevanceMap = programGetRelevanceMap.createCLKernel("kernelGetRelevanceMap");

                argn = 0;
                kernelGetRelevanceMap.setArg(argn++, clRefPixels);
                kernelGetRelevanceMap.setArg(argn++, clRelevanceMap);

                // Calculate
                queue.put2DRangeKernel(kernelGetRelevanceMap, 0, 0, w, h, 0, 0);
                queue.finish();

                // Read the relevance map back from the device
                queue.putReadBuffer(clRelevanceMap, true);
                for (int y=0; y<h; y++) {
                    for (int x=0; x<w; x++) {
                        relevanceMap[y*w+x] = clRelevanceMap.getBuffer().get(y*w+x);
                    }
                }
                queue.finish();

                // Calculate mean noise variance
                float noiseMeanVar = 0.0f;
                float n = 0.0f;
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        noiseMeanVar += relevanceMap[j*w+i];
                        n += 1.0f;
                    }
                }
                noiseMeanVar /= n;

                // Filter out irrelevant regions
                for(int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if(relevanceMap[j*w+i] <= noiseMeanVar*filterConstant) {
                            pearsonMap[j*w+i] = 0.0f;
                        }
                    }
                }

                if(normalizeOutput){
                    // ----------------------------------------------------------------------- //
                    // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
                    // ----------------------------------------------------------------------- //

                    // Find min and max within the relevance mask
                    float pearsonMin = Float.MAX_VALUE;
                    float pearsonMax = -Float.MAX_VALUE;

                    for (int j = bRH; j < h - bRH; j++) {
                        for (int i = bRW; i < w - bRW; i++) {
                            if (relevanceMap[j * w + i] > noiseMeanVar * filterConstant) {
                                float pixelValue = pearsonMap[j * w + i];
                                if (pixelValue > pearsonMax) {
                                    pearsonMax = pixelValue;
                                }
                                if (pixelValue < pearsonMin) {
                                    pearsonMin = pixelValue;
                                }
                            }
                        }
                    }

                    // Remap pixels
                    for (int j = bRH; j < h - bRH; j++) {
                        for (int i = bRW; i < w - bRW; i++) {
                            if (relevanceMap[j * w + i] > noiseMeanVar * filterConstant) {
                                pearsonMap[j * w + i] = (pearsonMap[j * w + i] - pearsonMin) / (pearsonMax - pearsonMin);
                            }
                        }
                    }
                }

            }else if(normalizeOutput){

                // -------------------------- //
                // ---- Normalize output ---- //
                // -------------------------- //

                // Find min and max
                float pearsonMin = Float.MAX_VALUE;
                float pearsonMax = -Float.MAX_VALUE;

                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        float pixelValue = pearsonMap[j*w+i];
                        if (pixelValue > pearsonMax) {
                            pearsonMax = pixelValue;
                        }
                        if (pixelValue < pearsonMin) {
                            pearsonMin = pixelValue;
                        }
                    }
                }

                // Remap pixels
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        pearsonMap[j*w+i] = (pearsonMap[j * w + i] - pearsonMin) / (pearsonMax - pearsonMin);
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMap);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }

        if(metric == metrics[1]) { // Absolute Difference of Standard Deviations
            showStatus("Calculating Absolute Difference of Standard Deviations...");

            // Build OpenCL program
            String programStringGetSynthPatchDiffStd = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchDiffStd.cl");
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$WIDTH$", "" + w);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$HEIGHT$", "" + h);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BRW$", "" + bRW);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$BRH$", "" + bRH);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$PATCH_STD$", "" + patchStdDev);
            programStringGetSynthPatchDiffStd = replaceFirst(programStringGetSynthPatchDiffStd, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchDiffStd = context.createProgram(programStringGetSynthPatchDiffStd).build();

            // Fill OpenCL buffers
            float[] diffStdMap = new float[wh];
            clDiffStdMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clDiffStdMap, diffStdMap);

            // Create kernel and set args
            kernelGetSynthPatchDiffStd = programGetSynthPatchDiffStd.createCLKernel("kernelGetSynthPatchDiffStd");

            argn = 0;
            kernelGetSynthPatchDiffStd.setArg(argn++, clLocalStds);
            kernelGetSynthPatchDiffStd.setArg(argn++, clDiffStdMap);

            // Calculate absolute difference of StdDevs
            queue.putWriteBuffer(clDiffStdMap, true);
            queue.put2DRangeKernel(kernelGetSynthPatchDiffStd, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's coefficients back from the GPU
            queue.putReadBuffer(clDiffStdMap, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    diffStdMap[y*w+x] = clDiffStdMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetSynthPatchDiffStd.release();
            clDiffStdMap.release();
            programGetSynthPatchDiffStd.release();

            // --------------------------------------- //
            // ---- Filter out irrelevant regions ---- //
            // --------------------------------------- //
            //TODO: SHOULDNT THIS BE 3% PRIME ASIA?
            // NOTE: THIS KERNEL IS THE SAME AS THE LOCAL STDS BUT WITHOUT NORMALIZING THE PATCHES. WE CAN USE THE SAME BUFFERS
            float[] diffStdMapNorm = new float[wh];

            if(filterConstant>0.0f) {
                showStatus("Calculating relevance map...");

                // Create OpenCL program
                String programStringGetRelevanceMap = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetRelevanceMap.cl");
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$WIDTH$", "" + w);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$HEIGHT$", "" + h);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$PATCH_SIZE$", "" + patchSize);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRW$", "" + bRW);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRH$", "" + bRH);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$EPSILON$", "" + EPSILON);
                programGetRelevanceMap = context.createProgram(programStringGetRelevanceMap).build();

                // Create and fill buffers
                float[] relevanceMap = new float[wh];
                clRelevanceMap = context.createFloatBuffer(wh, READ_WRITE);
                fillBufferWithFloatArray(clRelevanceMap, relevanceMap);
                queue.putWriteBuffer(clRelevanceMap, true);
                queue.finish();

                // Create OpenCL kernel and set args
                kernelGetRelevanceMap = programGetRelevanceMap.createCLKernel("kernelGetRelevanceMap");

                argn = 0;
                kernelGetRelevanceMap.setArg(argn++, clRefPixels);
                kernelGetRelevanceMap.setArg(argn++, clRelevanceMap);

                // Calculate
                queue.put2DRangeKernel(kernelGetRelevanceMap, 0, 0, w, h, 0, 0);
                queue.finish();

                // Read the relevance map back from the device
                queue.putReadBuffer(clRelevanceMap, true);
                for (int y=0; y<h; y++) {
                    for (int x=0; x<w; x++) {
                        relevanceMap[y*w+x] = clRelevanceMap.getBuffer().get(y*w+x);
                    }
                }
                queue.finish();

                // Calculate mean noise variance
                float noiseMeanVar = 0.0f;
                float n = 0.0f;
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        noiseMeanVar += relevanceMap[j*w+i];
                        n += 1.0f;
                    }
                }
                noiseMeanVar /= n;

                // Filter out irrelevant regions
                for(int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if(relevanceMap[j*w+i] <= noiseMeanVar*filterConstant) {
                            diffStdMap[j*w+i] = 0.0f;
                        }
                    }
                }

                // ----------------------------------------------------------------------- //
                // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
                // ----------------------------------------------------------------------- //

                // Find min and max within the relevance mask
                float diffStdMin = Float.MAX_VALUE;
                float diffStdMax = -Float.MAX_VALUE;

                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if (relevanceMap[j*w+i]>noiseMeanVar*filterConstant) {
                            float pixelValue = diffStdMap[j*w+i];
                            if (pixelValue > diffStdMax) {
                                diffStdMax = pixelValue;
                            }
                            if (pixelValue < diffStdMin) {
                                diffStdMin = pixelValue;
                            }
                        }
                    }
                }

                // Remap pixels
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if (relevanceMap[j*w+i]>noiseMeanVar*filterConstant) {
                            diffStdMapNorm[j*w+i] = (diffStdMap[j*w+i]-diffStdMin) / (diffStdMax-diffStdMin);
                        }
                    }
                }
            }else{

                // -------------------------- //
                // ---- Normalize output ---- //
                // -------------------------- //

                // Find min and max
                float diffStdMin = Float.MAX_VALUE;
                float diffStdMax = -Float.MAX_VALUE;

                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        float pixelValue = diffStdMap[j*w+i];
                        if (pixelValue > diffStdMax) {
                            diffStdMax = pixelValue;
                        }
                        if (pixelValue < diffStdMin) {
                            diffStdMin = pixelValue;
                        }
                    }
                }

                // Remap pixels
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        diffStdMapNorm[j*w+i] = (diffStdMap[j*w+i]-diffStdMin) / (diffStdMax - diffStdMin);
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            FloatProcessor fp1 = new FloatProcessor(w, h, diffStdMapNorm);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }

        if(metric == metrics[2]) { // Hu moments
            // Create OpenCL program
            String programStringGetSynthPatchHu = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchHu.cl");
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$WIDTH$", "" + w);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$HEIGHT$", "" + h);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$PATCH_SIZE$", "" + patchSize);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$BW$", "" + bW);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$BH$", "" + bH);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$BRW$", "" + bRW);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$BRH$", "" + bRH);
            programStringGetSynthPatchHu = replaceFirst(programStringGetSynthPatchHu, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchHu = context.createProgram(programStringGetSynthPatchHu).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            float[] huMap = new float[wh];
            clHuMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clHuMap, huMap);

            // Create OpenCL kernel and set args
            kernelGetSynthPatchHu = programGetSynthPatchHu.createCLKernel("kernelGetSynthPatchHu");

            argn = 0;
            kernelGetSynthPatchHu.setArg(argn++, clPatchPixels);
            kernelGetSynthPatchHu.setArg(argn++, clRefPixels);
            kernelGetSynthPatchHu.setArg(argn++, clLocalMeans);
            kernelGetSynthPatchHu.setArg(argn++, clHuMap);

            // Calculate
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clHuMap, true);
            queue.finish();

            // Enqueue kernel
            queue.put2DRangeKernel(kernelGetSynthPatchHu, 0, 0, w, h, 0, 0);

            // Read results back from the device
            queue.putReadBuffer(clHuMap, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    huMap[y*w+x] = clHuMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release memory
            kernelGetSynthPatchHu.release();
            clPatchPixels.release();
            clHuMap.release();
            programGetSynthPatchHu.release();


            // --------------------------------------- //
            // ---- Filter out irrelevant regions ---- //
            // --------------------------------------- //

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
                    if(localVars[j*w+i]<=noiseMeanVar*filterConstant){
                        huMap[j*w+i] = 0.0f;
                    }
                }
            }


            // ----------------------------------------------------------------------- //
            // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
            // ----------------------------------------------------------------------- //

            // Find min and max inside the relevance mask
            float huMin = 0.0f;
            float huMax = 0.0f;

            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    if(localVars[j*w+i]>noiseMeanVar*filterConstant) {
                        float pixelValue = huMap[j * w + i];
                        if (pixelValue > huMax) {
                            huMax = pixelValue;
                        }
                        if (pixelValue < huMin) {
                            huMin = pixelValue;
                        }
                    }
                }
            }

            // Remap pixels
            float[] huMapNorm = new float[wh];
            for(int j=bRH; j<h-bRH; j++){
                for(int i=bRW; i<w-bRW; i++){
                    if(localVars[j*w+i]>noiseMeanVar*filterConstant) {
                        huMapNorm[j*w+i] = (huMap[j*w+i] - huMin) / (huMax - huMin);
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            FloatProcessor fp1 = new FloatProcessor(w, h, huMapNorm);
            ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
            imp1.show();
        }

        if(metric == metrics[3]) { // mSSIM (i.e., SSIM without the luminance component)
            showStatus("Calculating mSSIM...");

            // Build OpenCL program
            String programStringGetSynthPatchSsim = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetSynthPatchSsim.cl");
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$WIDTH$", "" + w);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$HEIGHT$", "" + h);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$PATCH_SIZE$", "" + patchSize);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$BW$", "" + bW);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$BH$", "" + bH);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$BRW$", "" + bRW);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$BRH$", "" + bRH);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$PATCH_MEAN$", "" + patchMeanFloat);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$PATCH_STD$", "" + patchStdDev);
            programStringGetSynthPatchSsim = replaceFirst(programStringGetSynthPatchSsim, "$EPSILON$", "" + EPSILON);
            programGetSynthPatchSsim = context.createProgram(programStringGetSynthPatchSsim).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            float[] ssimMap = new float[wh];
            clSsimMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clSsimMap, ssimMap);

            // Create kernel and set args
            kernelGetSynthPatchSsim = programGetSynthPatchSsim.createCLKernel("kernelGetSynthPatchSsim");

            argn = 0;
            kernelGetSynthPatchSsim.setArg(argn++, clPatchPixels);
            kernelGetSynthPatchSsim.setArg(argn++, clRefPixels);
            //kernelGetSynthPatchSsim.setArg(argn++, clGaussianWindow);
            kernelGetSynthPatchSsim.setArg(argn++, clLocalMeans);
            kernelGetSynthPatchSsim.setArg(argn++, clLocalStds);
            kernelGetSynthPatchSsim.setArg(argn++, clSsimMap);

            // Calculate Pearson's correlation coefficient (reference patch vs. all)
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clSsimMap, true);
            queue.put2DRangeKernel(kernelGetSynthPatchSsim, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read Pearson's coefficients back from the GPU
            queue.putReadBuffer(clSsimMap, true);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    ssimMap[y*w+x] = clSsimMap.getBuffer().get(y*w+x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetSynthPatchSsim.release();
            clPatchPixels.release();
            clSsimMap.release();
            programGetSynthPatchSsim.release();


            // --------------------------------------- //
            // ---- Filter out irrelevant regions ---- //
            // --------------------------------------- //
            //TODO: SHOULDNT THIS BE 3% PRIME ASIA?
            // NOTE: THIS KERNEL IS THE SAME AS THE LOCAL STDS BUT WITHOUT NORMALIZING THE PATCHES. WE CAN USE THE SAME BUFFERS
            float[] ssimMapNorm = new float[wh];

            if(filterConstant!=0.0f) {
                showStatus("Calculating relevance map...");

                // Create OpenCL program
                String programStringGetRelevanceMap = getResourceAsString(ArtificialBlockRedundancy_.class, "kernelGetRelevanceMap.cl");
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$WIDTH$", "" + w);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$HEIGHT$", "" + h);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$PATCH_SIZE$", "" + patchSize);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRW$", "" + bRW);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$BRH$", "" + bRH);
                programStringGetRelevanceMap = replaceFirst(programStringGetRelevanceMap, "$EPSILON$", "" + EPSILON);
                programGetRelevanceMap = context.createProgram(programStringGetRelevanceMap).build();

                // Create and fill buffers
                float[] relevanceMap = new float[wh];
                clRelevanceMap = context.createFloatBuffer(wh, READ_WRITE);
                fillBufferWithFloatArray(clRelevanceMap, relevanceMap);
                queue.putWriteBuffer(clRelevanceMap, true);
                queue.finish();

                // Create OpenCL kernel and set args
                kernelGetRelevanceMap = programGetRelevanceMap.createCLKernel("kernelGetRelevanceMap");

                argn = 0;
                kernelGetRelevanceMap.setArg(argn++, clRefPixels);
                kernelGetRelevanceMap.setArg(argn++, clRelevanceMap);

                // Calculate
                queue.put2DRangeKernel(kernelGetRelevanceMap, 0, 0, w, h, 0, 0);
                queue.finish();

                // Read the relevance map back from the device
                queue.putReadBuffer(clRelevanceMap, true);
                for (int y=0; y<h; y++) {
                    for (int x=0; x<w; x++) {
                        relevanceMap[y*w+x] = clRelevanceMap.getBuffer().get(y*w+x);
                    }
                }
                queue.finish();

                // Calculate mean noise variance
                float noiseMeanVar = 0.0f;
                float n = 0.0f;
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        noiseMeanVar += relevanceMap[j*w+i];
                        n += 1.0f;
                    }
                }
                noiseMeanVar /= n;

                // Filter out irrelevant regions
                for(int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if(relevanceMap[j*w+i] <= noiseMeanVar*filterConstant) {
                            ssimMap[j*w+i] = 0.0f;
                        }
                    }
                }

                // ----------------------------------------------------------------------- //
                // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
                // ----------------------------------------------------------------------- //

                // Find min and max within the relevance mask
                float ssimMin = Float.MAX_VALUE;
                float ssimMax = -Float.MAX_VALUE;

                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if (relevanceMap[j*w+i]>noiseMeanVar*filterConstant) {
                            float pixelValue = ssimMap[j*w+i];
                            if (pixelValue > ssimMax) {
                                ssimMax = pixelValue;
                            }
                            if (pixelValue < ssimMin) {
                                ssimMin = pixelValue;
                            }
                        }
                    }
                }

                // Remap pixels
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        if (relevanceMap[j*w+i]>noiseMeanVar*filterConstant) {
                            ssimMapNorm[j*w+i] = (ssimMap[j*w+i]-ssimMin) / (ssimMax-ssimMin);
                        }
                    }
                }
            }else{

                // -------------------------- //
                // ---- Normalize output ---- //
                // -------------------------- //

                // Find min and max
                float ssimMin = Float.MAX_VALUE;
                float ssimMax = -Float.MAX_VALUE;

                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        float pixelValue = ssimMap[j*w+i];
                        if (pixelValue > ssimMax) {
                            ssimMax = pixelValue;
                        }
                        if (pixelValue < ssimMin) {
                            ssimMin = pixelValue;
                        }
                    }
                }

                // Remap pixels
                for (int j=bRH; j<h-bRH; j++) {
                    for (int i=bRW; i<w-bRW; i++) {
                        ssimMapNorm[j*w+i] = (ssimMap[j * w + i] - ssimMin) / (ssimMax - ssimMin);
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            FloatProcessor fp1 = new FloatProcessor(w, h, ssimMapNorm);
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
        float[] minMax = {Float.MAX_VALUE, -Float.MAX_VALUE};

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

    public static float[] normalize01(float[] inputArr, int w, int h, int bRW, int bRH, float min, float max, float EPSILON){
        float[] output = new float[w*h];
        for(int j=bRH; j<h-bRH; j++){
            for(int i=bRW; i<w-bRW; i++){
                output[j*w+i] = (inputArr[j*w+i]-min)/(max-min+EPSILON);
            }
        }
        return output;
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

