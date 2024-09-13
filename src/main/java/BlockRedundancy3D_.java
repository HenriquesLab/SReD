/**
 *
 * Returns a 3D repetition map where each pixel value represents the repetition score between the block centered around that pixel and a reference block.
 *
 * @author Afonso Mendes
 * @version 2023.11.01
 *
 */

import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.measure.Calibration;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;

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


public class BlockRedundancy3D_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans3D, programGetSynthPatchDiffStd, programGetPatchPearson3D,
            programGetSynthPatchHu, programGetSynthPatchSsim, programGetRelevanceMap3D;

    static private CLKernel kernelGetPatchMeans3D, kernelGetSynthPatchDiffStd, kernelGetPatchPearson3D,
            kernelGetSynthPatchHu, kernelGetSynthPatchSsim, kernelGetRelevanceMap3D;

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
        int nImages = getImageCount();
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
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Block Repetition (3D)");
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

        // Get ImagePlus and ImageStack
        ImagePlus imp = WindowManager.getImage(patchID);
        if (imp == null) {
            IJ.error("Patch image not found. Try again.");
            return;
        }

        ImageStack ims = imp.getStack();

        // Get patch dimensions
        int bW = ims.getWidth(); // Patch width
        int bH = ims.getHeight(); // Patch height
        int bZ = ims.getSize(); // Patch depth

        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int bRZ = bZ/2; // Patch radius (z-axis)

        // Check if patch dimensions are odd, otherwise kill program
        if (bW % 2 == 0 || bH % 2 == 0 || bZ % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3x3 or 5x5x5). Please try again.");
            return;
        }

        // Check if patch has at least 3 slices, otherwise kill program
        if (bZ < 3) {
            IJ.error("Patch must have at least 3 slices. Please try again.");
            return;
        }


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        // Get ImagePlus and ImageStack
        ImagePlus imp0 = WindowManager.getImage(imgID);
        if (imp0 == null) {
            IJ.error("Image not found. Try again.");
            return;
        }

        ImageStack ims0 = imp0.getStack();

        // Get calibration parameters
        Calibration calibration = imp.getCalibration();

        // Get image dimensions
        int w = ims0.getWidth();
        int h = ims0.getHeight();
        int z = ims0.getSize();
        int wh = w*h;
        int whz = w*h*z;

        // Check if image has at least 3 slices, otherwise kill program
        if (z < 3) {
            IJ.error("Image must have at least 3 slices. Please try again.");
            return;
        }

        // Check if patch doesn't have more slices than the image, otherwise kill program
        if (bZ > z) {
            IJ.error("Patch must have at least 3 slices. Please try again.");
            return;
        }

        //int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        // Patch
        IJ.log("Stabilizing noise variance of the patch...");

        float[][] patchPixels = new float[bZ][bW*bH];
        for(int n=0; n<bZ; n++){
            for(int y=0; y<bH; y++) {
                for(int x=0; x<bW; x++) {
                    patchPixels[n][y*bW+x] = ims.getProcessor(n+1).convertToFloatProcessor().getf(x,y);
                }
            }
        }

        GATMinimizer3D minimizer = new GATMinimizer3D(patchPixels, bW, bH, bZ, 0, 100, 0);
        minimizer.run();

        for(int n=0; n<bZ; n++){
            patchPixels[n] = VarianceStabilisingTransform3D_.getGAT(patchPixels[n], minimizer.gain, minimizer.sigma, minimizer.offset);
            ims.setProcessor(new FloatProcessor(bW, bH, patchPixels[n]), n+1);
        }

        // Image
        IJ.log("Stabilizing noise variance of the image...");

        float[][] refPixels = new float[z][wh];
        for(int n=0; n<z; n++) {
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    refPixels[n][y*w+x] = ims0.getProcessor(n+1).convertToFloatProcessor().getf(x,y);
                }
            }
        }

        minimizer = new GATMinimizer3D(refPixels, w, h, z, 0, 100, 0);
        minimizer.run();

        for(int n=0; n<z; n++) {
            refPixels[n] = VarianceStabilisingTransform3D_.getGAT(refPixels[n], minimizer.gain, minimizer.sigma, minimizer.offset);
        }


        // --------------------------------- //
        // ---- Process reference patch ---- //
        // --------------------------------- //

        // Get final patch size (after removing pixels outside the sphere/ellipsoid)
        int patchSize = 0;
        for(int n=0; n<bZ; n++) {
            for (int y = 0; y < bH; y++) {
                for (int x = 0; x < bW; x++) {
                    float dx = (float) (x - bRW);
                    float dy = (float) (y - bRH);
                    float dz = (float) (n - bRZ);
                    if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) + ((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f) {
                        patchSize++;
                    }
                }
            }
        }

        // Convert patch to double type (keeping only the pixels inside the sphere/ellipsoid)
        // Note: The 3D patch is flattened to 1D
        double[] patchPixelsDouble = new double[patchSize];
        int index = 0;
        for(int n=0; n<bZ; n++) {
            for (int y = 0; y < bH; y++) {
                for (int x = 0; x < bW; x++) {
                    float dx = (float) (x - bRW);
                    float dy = (float) (y - bRH);
                    float dz = (float) (n - bRZ);
                    if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) + ((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f) {
                        patchPixelsDouble[index] = patchPixels[n][y*bW+x];
                        index++;
                    }
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
            patchPixelsDouble[i] = (patchPixelsDouble[i] - patchMin) / (patchMax - patchMin + EPSILON);
            patchMean += patchPixelsDouble[i];
        }
        patchMean /= (double) patchSize;

        // Subtract mean
        for(int i=0; i<patchSize; i++){
            patchPixelsDouble[i] -= patchMean;
        }

        // Normalize again
        patchMin = Double.MAX_VALUE; // Initialize as a very large number
        patchMax = -Double.MAX_VALUE; // Initialize as a very small number

        for(int i=0; i<patchSize; i++){
            patchMin = min(patchMin, patchPixelsDouble[i]);
            patchMax = max(patchMax, patchPixelsDouble[i]);
        }

        for(int i=0; i<patchSize; i++){
            patchPixelsDouble[i] = (patchPixelsDouble[i] - patchMin) / (patchMax - patchMin + EPSILON);
        }

        // Typecast to float
        float[] patchPixelsFloat = new float[patchSize];
        for(int i=0; i<patchSize; i++){
            patchPixelsFloat[i] = (float) patchPixelsDouble[i];
        }

        // Calculate mean and standard deviation
        float patchMeanFloat = 0.0f;
        float patchStdDev = 0.0f;
        for(int i=0; i<patchSize; i++){
            patchMeanFloat += patchPixelsFloat[i];
            patchStdDev += (patchPixelsFloat[i]-(float)patchMean)*(patchPixelsFloat[i]-(float)patchMean);
        }

        patchMeanFloat /= (float) patchSize;
        patchStdDev = (float) sqrt(patchStdDev / (float) patchSize);


        // ----------------------- //
        // ---- Process image ---- //
        // ----------------------- //

        // Cast to double type and store as flattened 1D array
        double[] refPixelsDouble = new double[whz];
        for(int n=0; n<z; n++) {
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    refPixelsDouble[w*h*n+y*w+x]=refPixels[n][y*w+x];
                }
            }
        }

        // Get min and max
        double imgMin = Double.MAX_VALUE; // Initialize as a very large number
        double imgMax = -Double.MAX_VALUE; // Initialize as a very small number
        for(int i=0; i<whz; i++){
            double pixelValue = refPixelsDouble[i];
            if(pixelValue<imgMin){
                imgMin = pixelValue;
            }
            if(pixelValue>imgMax){
                imgMax = pixelValue;
            }
        }

        // Normalize
        for(int i=0; i<whz; i++){
            refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + EPSILON);
        }

        // Cast back to float
        float[] refPixelsFloat = new float[whz];
        for (int i=0; i<whz; i++) {
            refPixelsFloat[i] = (float) refPixelsDouble[i];
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
        for (int i=0; i<allDevices.length; i++) {
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

        // Create OpenCL command queue
        queue = chosenDevice.createCommandQueue();
        int elementCount = whz;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        IJ.log("Calculating block repetition...");


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(whz, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(whz, READ_WRITE);
        clLocalStds = context.createFloatBuffer(whz, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans3D = getResourceAsString(BlockRedundancy3D_.class, "kernelGetPatchMeans3D.cl");
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$WIDTH$", "" + w);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$HEIGHT$", "" + h);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$DEPTH$", "" + z);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$BRW$", "" + bRW);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$BRH$", "" + bRH);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$BRZ$", "" + bRZ);
        programStringGetPatchMeans3D = replaceFirst(programStringGetPatchMeans3D, "$EPSILON$", "" + EPSILON);
        programGetPatchMeans3D = context.createProgram(programStringGetPatchMeans3D).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixelsFloat);

        float[] localMeans = new float[whz];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[whz];
        fillBufferWithFloatArray(clLocalStds, localStds);

        // Create OpenCL kernel and set arguments
        kernelGetPatchMeans3D = programGetPatchMeans3D.createCLKernel("kernelGetPatchMeans3D");

        int argn = 0;
        kernelGetPatchMeans3D.setArg(argn++, clRefPixels);
        kernelGetPatchMeans3D.setArg(argn++, clLocalMeans);
        kernelGetPatchMeans3D.setArg(argn++, clLocalStds);

        // Write buffers
        queue.putWriteBuffer(clRefPixels, true);
        queue.putWriteBuffer(clLocalMeans, true);
        queue.putWriteBuffer(clLocalStds, true);

        // Calculate
        showStatus("Calculating local means...");
        queue.put3DRangeKernel(kernelGetPatchMeans3D, 0, 0, 0, w, h, z, 0, 0, 0);
        queue.finish();

        // Read the local means map back from the device
        queue.putReadBuffer(clLocalMeans, true);
        for(int n=0; n<z; n++){
            for(int y=0; y<h; y++){
                for(int x=0; x<w; x++){
                    localMeans[w*h*n+y*w+x] = clLocalMeans.getBuffer().get(w*h*n+y*w+x);
                }
            }
        }
        queue.finish();

        // Read the local stds map back from the device
        queue.putReadBuffer(clLocalStds, true);
        for(int n=0; n<z; n++){
            for(int y=0; y<h; y++){
                for(int x=0; x<w; x++){
                    localStds[w*h*n+y*w+x] = clLocalStds.getBuffer().get(w*h*n+y*w+x);
                }
            }
        }
        queue.finish();

        // Release memory
        kernelGetPatchMeans3D.release();
        programGetPatchMeans3D.release();


        // --------------------------------------------------------------- //
        // ---- Calculate block repetition map with the chosen metric ---- //
        // --------------------------------------------------------------- //

        if(metric == metrics[0]) { // Pearson correlation
            showStatus("Calculating Pearson correlations...");

            // Build OpenCL program
            String programStringGetPatchPearson3D = getResourceAsString(BlockRedundancy3D_.class, "kernelGetPatchPearson3D.cl");
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$WIDTH$", "" + w);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$HEIGHT$", "" + h);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$DEPTH$", "" + z);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BW$", "" + bW);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BH$", "" + bH);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BZ$", "" + bZ);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BRW$", "" + bRW);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BRH$", "" + bRH);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$BRZ$", "" + bRZ);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$PATCH_MEAN$", "" + patchMeanFloat);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$PATCH_STD$", "" + patchStdDev);
            programStringGetPatchPearson3D = replaceFirst(programStringGetPatchPearson3D, "$EPSILON$", "" + EPSILON);
            programGetPatchPearson3D = context.createProgram(programStringGetPatchPearson3D).build();

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            float[] pearsonMap = new float[whz];
            clPearsonMap = context.createFloatBuffer(whz, READ_WRITE);
            fillBufferWithFloatArray(clPearsonMap, pearsonMap);

            // Create kernel and set args
            kernelGetPatchPearson3D = programGetPatchPearson3D.createCLKernel("kernelGetPatchPearson3D");

            argn = 0;
            kernelGetPatchPearson3D.setArg(argn++, clPatchPixels);
            kernelGetPatchPearson3D.setArg(argn++, clRefPixels);
            kernelGetPatchPearson3D.setArg(argn++, clLocalMeans);
            kernelGetPatchPearson3D.setArg(argn++, clLocalStds);
            kernelGetPatchPearson3D.setArg(argn++, clPearsonMap);

            // Calculate Pearson's correlation coefficient (reference patch vs. all)
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clPearsonMap, true);
            queue.put3DRangeKernel(kernelGetPatchPearson3D,0,0,0, w, h, z,0,0,0);
            queue.finish();

            // Read Pearson's coefficients back from the GPU
            queue.putReadBuffer(clPearsonMap, true);

            for(int n=0; n<z; n++) {
                for (int y=0; y<h; y++) {
                    for (int x=0; x<w; x++) {
                        pearsonMap[w*h*n+y*w+x] = clPearsonMap.getBuffer().get(w*h*n+y*w+x);
                        queue.finish();
                    }
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetPatchPearson3D.release();
            clPatchPixels.release();
            clPearsonMap.release();
            programGetPatchPearson3D.release();


            // --------------------------------------- //
            // ---- Filter out irrelevant regions ---- //
            // --------------------------------------- //

            if(filterConstant>0.0f) {
                showStatus("Calculating relevance map...");

                // Create OpenCL program
                String programStringGetRelevanceMap3D = getResourceAsString(BlockRedundancy3D_.class, "kernelGetRelevanceMap3D.cl");
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$WIDTH$", "" + w);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$HEIGHT$", "" + h);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$DEPTH$", "" + z);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$PATCH_SIZE$", "" + patchSize);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$BRW$", "" + bRW);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$BRH$", "" + bRH);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$BRZ$", "" + bRZ);
                programStringGetRelevanceMap3D = replaceFirst(programStringGetRelevanceMap3D, "$EPSILON$", "" + EPSILON);
                programGetRelevanceMap3D = context.createProgram(programStringGetRelevanceMap3D).build();

                // Create and fill buffers
                float[] relevanceMap = new float[whz];
                clRelevanceMap = context.createFloatBuffer(wh, READ_WRITE);
                fillBufferWithFloatArray(clRelevanceMap, relevanceMap);
                queue.putWriteBuffer(clRelevanceMap, true);
                queue.finish();

                // Create OpenCL kernel and set args
                kernelGetRelevanceMap3D = programGetRelevanceMap3D.createCLKernel("kernelGetRelevanceMap3D");

                argn = 0;
                kernelGetRelevanceMap3D.setArg(argn++, clRefPixels);
                kernelGetRelevanceMap3D.setArg(argn++, clRelevanceMap);

                // Calculate
                queue.put3DRangeKernel(kernelGetRelevanceMap3D, 0, 0, 0, w, h, z, 0, 0, 0);
                queue.finish();

                // Read the relevance map back from the device
                queue.putReadBuffer(clRelevanceMap, true);
                for(int n=0; n<z; n++) {
                    for (int y=0; y<h; y++) {
                        for (int x=0; x<w; x++) {
                            relevanceMap[w*h*n+y*w+x] = clRelevanceMap.getBuffer().get(w*h*n+y*w+x);
                        }
                    }
                }
                queue.finish();

                // Calculate mean noise variance
                float noiseMeanVar = 0.0f;
                float numElements = 0.0f;
                for(int n=bRZ; n<z-bRZ; n++){
                    for (int j=bRH; j<h-bRH; j++) {
                        for (int i=bRW; i<w-bRW; i++) {
                            noiseMeanVar += relevanceMap[w*h*n+j*w+i];
                            numElements += 1.0f;
                        }
                    }
                }
                noiseMeanVar /= numElements;

                // Filter out irrelevant regions
                for(int n=bRZ; n<z-bRZ; n++){
                    for (int j=bRH; j<h-bRH; j++) {
                        for (int i=bRW; i<w-bRW; i++) {
                            if (relevanceMap[w*h*n+j*w+i] <= noiseMeanVar * filterConstant) {
                                pearsonMap[w*h*n+j*w+i] = 0.0f;
                            }
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

                    for(int n=bRZ; n<z-bRZ; n++){
                        for (int j=bRH; j<h-bRH; j++) {
                            for (int i=bRW; i<w-bRW; i++) {
                                if (relevanceMap[w*h*n+j*w+i] > noiseMeanVar * filterConstant) {
                                    float pixelValue = pearsonMap[w*h*n+j*w+i];
                                    if (pixelValue > pearsonMax) {
                                        pearsonMax = pixelValue;
                                    }
                                    if (pixelValue < pearsonMin) {
                                        pearsonMin = pixelValue;
                                    }
                                }
                            }
                        }
                    }

                    // Remap pixels
                    for(int n=bRZ; n<z-bRZ; n++){
                        for (int j=bRH; j<h-bRH; j++) {
                            for (int i=bRW; i<w-bRW; i++) {
                                if (relevanceMap[w*h*n+j*w+i] > noiseMeanVar * filterConstant) {
                                    pearsonMap[w*h*n+j*w+i] = (pearsonMap[w*h*n+j*w+i] - pearsonMin) / (pearsonMax - pearsonMin);
                                }
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

                for(int n=bRZ; n<z-bRZ; n++) {
                    for (int j=bRH; j<h-bRH; j++) {
                        for (int i=bRW; i<w-bRW; i++) {
                            float pixelValue = pearsonMap[w*h*n+j*w+i];
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
                for(int n=bRZ; n<z-bRZ; n++) {
                    for (int j=bRH; j<h-bRH; j++) {
                        for (int i=bRW; i<w-bRW; i++) {
                            pearsonMap[w*h*n+j*w+i] = (pearsonMap[w*h*n+j*w+i] - pearsonMin) / (pearsonMax - pearsonMin);
                        }
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            ImageStack imsFinal = new ImageStack(w, h, z);
            for(int n=0; n<z; n++){
                FloatProcessor temp = new FloatProcessor(w, h);
                for(int y=0; y<h; y++){
                    for(int x=0; x<w; x++){
                        temp.setf(x, y, pearsonMap[w*h*n+y*w+x]);
                    }
                }
                imsFinal.setProcessor(temp, n+1);
            }
            ImagePlus impFinal = new ImagePlus("Block Repetition Map", imsFinal);
            impFinal.setCalibration(calibration);
            impFinal.show();
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

    public static String replaceFirst(String source, String target, String replacement) {
        int index = source.indexOf(target);
        if (index == -1) {
            return source;
        }

        return source.substring(0, index)
                .concat(replacement)
                .concat(source.substring(index+target.length()));
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

