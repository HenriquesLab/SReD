//TODO: Filling buffer with a patch writes wrong values. Currently the kernels are reading the reference patch from the image buffer based on the patch position. Try to fix this to use a patch written in a buffer.

import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.measure.Calibration;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;
import static java.lang.Math.*;
import static nanoj.core2.NanoJCL.replaceFirst;


public class BlockRedundancy2DT_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetPatchPearson, programGetRelevanceMap;

    static private CLKernel kernelGetPatchMeans, kernelGetPatchPearson, kernelGetRelevanceMap;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clPatchPixels, clRefPixels, clLocalMeans, clLocalStds, clPearsonMap, clRelevanceMap;

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
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Block Redundancy (2D)");
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

        // Get ImagePlus
        ImagePlus impPatch = WindowManager.getImage(patchID);
        if(impPatch == null) {
            IJ.error("Patch image not found. Please open an image and try again.");
            return;
        }

        // Get ImageProcessor
        ImageProcessor ipPatch = impPatch.getProcessor();
        FloatProcessor fpPatch = ipPatch.convertToFloatProcessor();
        float[] patchPixels = (float[]) fpPatch.getPixels();

        // Get patch dimensions
        int bW = impPatch.getWidth(); // Patch width
        int bH = impPatch.getHeight(); // Patch height

        int bRW = bW / 2; // Patch radius (x-axis)
        int bRH = bH / 2; // Patch radius (y-axis)

        // Check if patch dimensions are odd, otherwise kill program
        if(bW % 2 == 0 || bH % 2 == 0){
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        // Get ImagePlus and ImageStack
        ImagePlus impImage = WindowManager.getCurrentImage();
        if (impImage == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        ImageStack imsImage = impImage.getStack();

        // Get image dimensions
        int w = impImage.getWidth();
        int h = impImage.getHeight();
        int wh = w * h;
        int nFrames = impImage.getNFrames();

        if (nFrames < 2) {
            IJ.error("A timelapse was not detected. For single 2D images, please use the \"Block Redundancy (2D)\" plugin.");
            return;
        }

        // Get calibration parameters
        Calibration calibration = impImage.getCalibration();


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        // Patch
        IJ.log("Stabilising noise variance of the patch...");
        GATMinimizer2D minimizer2D = new GATMinimizer2D(patchPixels, bW, bH, 0, 100, 0);
        minimizer2D.run();
        patchPixels = VarianceStabilisingTransform2D_.getGAT(patchPixels, minimizer2D.gain, minimizer2D.sigma, minimizer2D.offset);
        IJ.log("Done.");

        // Image
        IJ.log("Stabilising noise variance of the image...");
        float[][] refPixels = new float[nFrames][wh];
        for(int f=0; f<nFrames; f++){
            for(int y=0; y<h; y++){
                for(int x=0; x<w; x++){
                    refPixels[f][y*w+x] = imsImage.getProcessor(f+1).getf(x,y);
                }
            }
        }

        GATMinimizer3D minimizer3D = new GATMinimizer3D(refPixels, w, h, nFrames, 0, 100, 0);
        minimizer3D.run();

        for(int f=0; f<nFrames; f++){
            refPixels[f] = VarianceStabilisingTransform3D_.getGAT(refPixels[f], minimizer3D.gain, minimizer3D.sigma, minimizer3D.offset);
        }


        // --------------------------------- //
        // ---- Process reference patch ---- //
        // --------------------------------- //

        // Get final patch size (after removing pixels outside the sphere/ellipsoid)
        int patchSize = 0;
        for (int y=0; y<bH; y++) {
            for (int x=0; x<bW; x++) {
                float dx = (float) (x-bRW);
                float dy = (float) (y-bRH);
                if (((dx*dx)/(float)(bRW*bRW)) + ((dy*dy)/(float)(bRH*bRH))<= 1.0f) {
                    patchSize++;
                }
            }
        }

        // Convert patch to "double" type (keeping only the pixels within the inbound circle/ellipse)
        double[] patchPixelsDouble = new double[patchSize];
        int index = 0;
        for(int j=0; j<bH; j++){
            for (int i=0; i<bW; i++) {
                float dx = (float)(i-bRW);
                float dy = (float)(j-bRH);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                    patchPixelsDouble[index] = (double)patchPixels[j*bW+i];
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
        for(int i=0; i<patchSize; i++){
            patchPixelsFloat[i] = (float)patchPixelsDouble[i];
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

        // Cast to double type and store as flattened 1D array
        double[][] refPixelsDouble = new double[nFrames][wh];
        for(int f=0; f<nFrames; f++) {
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    refPixelsDouble[f][y*w+x]=refPixels[f][y*w+x];
                }
            }
        }

        // MinMax normalization (of each frame separately)
        double imgMin = Double.MAX_VALUE; // Initialize as a very large number
        double imgMax = -Double.MAX_VALUE; // Initialize as a very small number
        for(int f=0; f<nFrames; f++) {

            // Get min and max
            imgMin = Double.MAX_VALUE; // Initialize as a very large number
            imgMax = -Double.MAX_VALUE; // Initialize as a very small number
            for (int i=0; i<wh; i++) {
                double pixelValue = refPixelsDouble[f][i];
                if (pixelValue < imgMin) {
                    imgMin = pixelValue;
                }
                if (pixelValue > imgMax) {
                    imgMax = pixelValue;
                }
            }
            // Remap pixels
            for (int i=0; i<wh; i++) {
                refPixelsDouble[f][i] = (refPixelsDouble[f][i]-imgMin)/(imgMax-imgMin+EPSILON);
            }
        }

        // Cast back to float
        float[][] refPixelsFloat = new float[nFrames][wh];
        for(int f=0; f<nFrames; f++) {
            for (int i=0; i<wh; i++) {
                refPixelsFloat[f][i] = (float) refPixelsDouble[f][i];
            }
        }


        // --------------------------- //
        // ---- Initialise OpenCL ---- //
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

        // Create context
        context = CLContext.create(clPlatformMaxFlop);

        // Choose the best device (i.e., Filter out CPUs if GPUs are available (FLOPS calculation was giving higher ratings to CPU vs. GPU))
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
        int elementCount = wh;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);


        // -------------------------------------------------------- //
        // ---- Calculate repetition for each frame separately ---- //
        // -------------------------------------------------------- //

        // Create stack to store final results
        ImageStack imsFinal = new ImageStack(w, h, nFrames);

        // Process frames
        for(int f=0; f<nFrames; f++){
            IJ.log("Processing frame " + (f+1) + "/" + nFrames);
            IJ.showStatus("Processing frame " + (f+1) + "/" + nFrames);
            IJ.showProgress(f, nFrames);


            // ------------------------------- //
            // ---- Calculate local means ---- //
            // ------------------------------- //

            // Create OpenCL buffers
            clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
            clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
            clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

            // Create OpenCL program
            String programStringGetPatchMeans = getResourceAsString(BlockRedundancy2DT_.class, "kernelGetPatchMeans2D.cl");
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
            programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$EPSILON$", "" + EPSILON);
            programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

            // Fill OpenCL buffers
            fillBufferWithFloatArray(clRefPixels, refPixelsFloat[f]);

            float[] localMeans = new float[wh];
            fillBufferWithFloatArray(clLocalMeans, localMeans);

            float[] localStds = new float[wh];
            fillBufferWithFloatArray(clLocalStds, localStds);

            // Create OpenCL kernel and set args
            kernelGetPatchMeans = programGetPatchMeans.createCLKernel("kernelGetPatchMeans2D");

            int argn = 0;
            kernelGetPatchMeans.setArg(argn++, clRefPixels);
            kernelGetPatchMeans.setArg(argn++, clLocalMeans);
            kernelGetPatchMeans.setArg(argn++, clLocalStds);

            // Calculate
            queue.putWriteBuffer(clRefPixels, true);
            queue.putWriteBuffer(clLocalMeans, true);
            queue.putWriteBuffer(clLocalStds, true);

            IJ.showStatus("Calculating local means... Frame " + (f+1) + "/" + nFrames);
            IJ.showProgress(f, nFrames);

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
                IJ.showStatus("Calculating Pearson's correlations... Frame " + (f+1) + "/" + nFrames);
                IJ.showProgress(f, nFrames);

                // Build OpenCL program
                String programStringGetPatchPearson = getResourceAsString(BlockRedundancy2DT_.class, "kernelGetPatchPearson2D.cl");
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$WIDTH$", "" + w);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$HEIGHT$", "" + h);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_SIZE$", "" + patchSize);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BW$", "" + bW);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BH$", "" + bH);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRW$", "" + bRW);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$BRH$", "" + bRH);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_MEAN$", "" + patchMeanFloat);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$PATCH_STD$", "" + patchStdDev);
                programStringGetPatchPearson = replaceFirst(programStringGetPatchPearson, "$EPSILON$", "" + EPSILON);
                programGetPatchPearson = context.createProgram(programStringGetPatchPearson).build();

                // Fill OpenCL buffers
                clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
                fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

                float[] pearsonMap = new float[wh];
                clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
                fillBufferWithFloatArray(clPearsonMap, pearsonMap);

                // Create kernel and set args
                kernelGetPatchPearson = programGetPatchPearson.createCLKernel("kernelGetPatchPearson2D");

                argn = 0;
                kernelGetPatchPearson.setArg(argn++, clPatchPixels);
                kernelGetPatchPearson.setArg(argn++, clRefPixels);
                kernelGetPatchPearson.setArg(argn++, clLocalMeans);
                kernelGetPatchPearson.setArg(argn++, clLocalStds);
                kernelGetPatchPearson.setArg(argn++, clPearsonMap);

                // Calculate Pearson's correlation coefficient (reference patch vs. all)
                queue.putWriteBuffer(clPatchPixels, true);
                queue.putWriteBuffer(clPearsonMap, true);
                queue.put2DRangeKernel(kernelGetPatchPearson, 0, 0, w, h, 0, 0);
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
                kernelGetPatchPearson.release();
                clPatchPixels.release();
                clPearsonMap.release();
                programGetPatchPearson.release();


                // --------------------------------------- //
                // ---- Filter out irrelevant regions ---- //
                // --------------------------------------- //
                //TODO: SHOULDNT THIS BE 3% PRIME ASIA?
                // NOTE: THIS KERNEL IS THE SAME AS THE LOCAL STDS BUT WITHOUT NORMALIZING THE PATCHES. WE CAN USE THE SAME BUFFERS

                if(filterConstant>0.0f) {
                    IJ.showStatus("Calculating relevance map... Frame " + (f+1) + "/" + nFrames);
                    IJ.showProgress(f, nFrames);

                    // Create OpenCL program
                    String programStringGetRelevanceMap = getResourceAsString(BlockRedundancy2DT_.class, "kernelGetRelevanceMap2D.cl");
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
                    kernelGetRelevanceMap = programGetRelevanceMap.createCLKernel("kernelGetRelevanceMap2D");

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

                    // Release resources
                    kernelGetRelevanceMap.release();
                    clRelevanceMap.release();
                    programGetRelevanceMap.release();

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


                // ----------------------- //
                // ---- Store results ---- //
                // ----------------------- //

                FloatProcessor fp1 = new FloatProcessor(w, h, pearsonMap);
                imsFinal.setProcessor(fp1, f+1);
            }

        }


        // ------------------------ //
        // ---- Display output ---- //
        // ------------------------ //

        ImagePlus impFinal = new ImagePlus("Block Redundancy Map", imsFinal);
        impFinal.setCalibration(calibration);
        impFinal.show();

        IJ.log("Releasing resources...");
        IJ.log("--------");

        // Release resources
        context.release();


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
}


