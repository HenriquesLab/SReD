/**
 *
 * This class calculates block repetition maps for 2D data.
 *
 * @author Afonso Mendes
 *
 **/

import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.LutLoader;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.LUT;

import java.awt.image.IndexColorModel;
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


public class BlockRedundancy2D_ implements PlugIn {

    // ------------------------ //
    // ---- OpenCL formats ---- //
    // ------------------------ //

    static private CLContext context;

    static private CLProgram programGetPatchMeans, programGetPatchCosineSim, programGetPatchDiffStd, programGetPatchPearson,
            programGetPatchHu, programGetPatchSsim, programGetRelevanceMap;

    static private CLKernel kernelGetPatchMeans, kernelGetPatchDiffStd, kernelGetPatchPearson,
            kernelGetPatchHu, kernelGetPatchSsim, kernelGetRelevanceMap, kernelGetPatchCosineSim;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clPatchPixels, clCosineSimMap, clDiffStdMap, clPearsonMap,
            clHuMap, clSsimMap, clRelevanceMap, clGaussianWindow;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;

        // Install SReD LUT
        installLUT.run();

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
        for (int i = 0; i < nImages; i++) {
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Define metric possibilities
        String[] metrics = new String[4];
        metrics[0] = "Pearson's R";
        metrics[1] = "Cosine similarity";
        metrics[2] = "SSIM";
        metrics[3] = "NRMSE (inverted)";

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Block Repetition (2D)");
        gd.addChoice("Block:", titles, titles[1]);
        gd.addChoice("Image:", titles, titles[0]);
        gd.addSlider("Filter constant: ", 0.0f, 5.0f, 0.0f, 0.1f);
        gd.addChoice("Metric:", metrics, metrics[0]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameters from dialog box
        String patchTitle = gd.getNextChoice();
        int patchID = 0;
        for (int i = 0; i < nImages; i++) {
            if (titles[i].equals(patchTitle)) { // .equals() instead of "==" required to run from macro
                patchID = ids[i];
            }
        }

        String imgTitle = gd.getNextChoice();
        int imgID = 0;
        for (int i = 0; i < nImages; i++) {
            if (titles[i].equals(imgTitle)) { // .equals() instead of "==" required to run from macro
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
            IJ.error("Block image not found. Try again.");
            return;
        }
        ImageProcessor ip = imp.getProcessor();
        FloatProcessor fp = ip.convertToFloatProcessor();
        float[] patchPixels = (float[]) fp.getPixels();
        int bW = fp.getWidth(); // Patch width
        int bH = fp.getHeight(); // Patch height

        // Check if patch dimensions are odd, otherwise kill program
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Block dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        // Calculate block radius
        int bRW = bW / 2; // Patch radius (x-axis)
        int bRH = bH / 2; // Patch radius (y-axis)


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
        int wh = w * h;
        //int sizeWithoutBorders = (w-bRW*2)*(h-bRH*2); // The area of the search field (= image without borders)


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        // Patch
        //IJ.log("Stabilising noise variance of the patch...");
        //GATMinimizer2D minimizer = new GATMinimizer2D(patchPixels, bW, bH, 0, 100, 0);
        //minimizer.run();
        //patchPixels = VarianceStabilisingTransform2D_.getGAT(patchPixels, minimizer.gain, minimizer.sigma, minimizer.offset);
        //IJ.log("Done.");

        // Image
        IJ.log("Stabilising noise variance of the image...");
        GATMinimizer2D minimizer = new GATMinimizer2D(refPixels, w, h, 0, 100, 0);
        minimizer.run();
        refPixels = VarianceStabilisingTransform2D_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);
        IJ.log("Done.");


        // --------------------------------- //
        // ---- Process reference block ---- //
        // --------------------------------- //

        // Get final block size (after removing pixels outside inbound circle/ellipse)
        int patchSize = 0;
        for (int j = 0; j < bH; j++) {
            for (int i = 0; i < bW; i++) {
                float dx = (float) (i - bRW);
                float dy = (float) (j - bRH);
                if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) <= 1.0f) {
                    patchSize++;
                }
            }
        }

        // Convert patch to "double" type (keeping only the pixels within the inbound circle/ellipse)
        double[] patchPixelsDouble = new double[patchSize];
        int index = 0;
        for (int j = 0; j < bH; j++) {
            for (int i = 0; i < bW; i++) {
                float dx = (float) (i - bRW);
                float dy = (float) (j - bRH);
                if (((dx * dx) / (float) (bRW * bRW)) + ((dy * dy) / (float) (bRH * bRH)) <= 1.0f) {
                    patchPixelsDouble[index] = (double) patchPixels[j * bW + i];
                    index++;
                }
            }
        }

        // Find min and max
        double patchMin = Double.MAX_VALUE; // Initialize as a very large number
        double patchMax = -Double.MAX_VALUE; // Initialize as a very small number

        for (int i = 0; i < patchSize; i++) {
            patchMin = min(patchMin, patchPixelsDouble[i]);
            patchMax = max(patchMax, patchPixelsDouble[i]);
        }

        // Normalize and calculate mean
        double patchMean = 0.0;
        for (int i = 0; i < patchSize; i++) {
            patchPixelsDouble[i] = (patchPixelsDouble[i] - patchMin) / (patchMax - patchMin + EPSILON);
            patchMean += patchPixelsDouble[i];

        }
        patchMean /= (double) patchSize;

        // Subtract mean
        for (int i = 0; i < patchSize; i++) {
            patchPixelsDouble[i] = patchPixelsDouble[i] - patchMean;
        }

        // Typecast back to float
        float[] patchPixelsFloat = new float[patchSize];
        for (int i = 0; i < patchSize; i++) {
            patchPixelsFloat[i] = (float) patchPixelsDouble[i];
        }

        // Calculate standard deviation
        float patchMeanFloat = (float) patchMean;
        double patchStdDev = 0.0;
        for (int i = 0; i < patchSize; i++) {
            patchStdDev += (patchPixelsDouble[i] - patchMean) * (patchPixelsDouble[i] - patchMean);
        }
        patchStdDev = (float) sqrt(patchStdDev / (patchSize - 1));


        // ----------------------- //
        // ---- Process image ---- //
        // ----------------------- //

        // Cast to "double" type
        double[] refPixelsDouble = new double[wh];
        for (int i = 0; i < wh; i++) {
            refPixelsDouble[i] = (double) refPixels[i];
        }

        // Get min and max
        double imgMin = Double.MAX_VALUE;
        double imgMax = -Double.MAX_VALUE;
        for (int i = 0; i < wh; i++) {
            double pixelValue = refPixelsDouble[i];
            if (pixelValue < imgMin) {
                imgMin = pixelValue;
            }
            if (pixelValue > imgMax) {
                imgMax = pixelValue;
            }
        }

        // Normalize
        for (int i = 0; i < wh; i++) {
            refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + EPSILON);
        }

        // Cast back to float
        for (int i = 0; i < wh; i++) {
            refPixels[i] = (float) refPixelsDouble[i];
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
                //IJ.log("--------");
                //IJ.log("Device name: " + clDevice.getName());
                //IJ.log("Device type: " + clDevice.getType());
                //IJ.log("Max clock: " + clDevice.getMaxClockFrequency() + " MHz");
                //IJ.log("Number of compute units: " + clDevice.getMaxComputeUnits());
                //IJ.log("Max work group size: " + clDevice.getMaxWorkGroupSize());
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
        if (useDevice) {
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
        int elementCount = w * h;
        int localWorkSize = min(chosenDevice.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        IJ.log("Calculating redundancy...");


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans = getResourceAsString(BlockRedundancy2D_.class, "kernelGetPatchMeans2D.cl");
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$WIDTH$", "" + w);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$HEIGHT$", "" + h);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$PATCH_SIZE$", "" + patchSize);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRW$", "" + bRW);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$BRH$", "" + bRH);
        programStringGetPatchMeans = replaceFirst(programStringGetPatchMeans, "$EPSILON$", "" + EPSILON);
        programGetPatchMeans = context.createProgram(programStringGetPatchMeans).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixels);

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

        showStatus("Calculating local means...");

        queue.put2DRangeKernel(kernelGetPatchMeans, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the local means map back from the device
        queue.putReadBuffer(clLocalMeans, true);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                localMeans[y * w + x] = clLocalMeans.getBuffer().get(y * w + x);
                queue.finish();

            }
        }

        // Read the local stds map back from the device
        queue.putReadBuffer(clLocalStds, true);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                localStds[y * w + x] = clLocalStds.getBuffer().get(y * w + x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchMeans.release();
        programGetPatchMeans.release();


        // --------------------------------------------------------------- //
        // ---- Calculate block repetition map with the chosen metric ---- //
        // --------------------------------------------------------------- //

        float[] repetitionMap = new float[wh]; // Array to store output repetition map

        if (metric == metrics[0]) { // Pearson correlation
            showStatus("Calculating Pearson correlations...");

            // Build OpenCL program
            String programStringGetPatchPearson = getResourceAsString(BlockRedundancy2D_.class, "kernelGetPatchPearson2D.cl");
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
            //System.out.println(programGetPatchPearson.getBuildLog()); // Print program build log to check for errors

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            clPearsonMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clPearsonMap, repetitionMap);

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
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    repetitionMap[y * w + x] = clPearsonMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetPatchPearson.release();
            clPatchPixels.release();
            clPearsonMap.release();
            programGetPatchPearson.release();
        }

        if (metric == metrics[1]) { // Cosine similarity
            showStatus("Calculating Cosine similarity...");

            // Build OpenCL program
            String programStringGetPatchCosineSim = getResourceAsString(BlockRedundancy2D_.class, "kernelGetPatchCosineSim2D.cl");
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$WIDTH$", "" + w);
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$HEIGHT$", "" + h);
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$BRW$", "" + bRW);
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$BRH$", "" + bRH);
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$PATCH_STD$", "" + patchStdDev);
            programStringGetPatchCosineSim = replaceFirst(programStringGetPatchCosineSim, "$EPSILON$", "" + EPSILON);
            programGetPatchCosineSim = context.createProgram(programStringGetPatchCosineSim).build();

            // Fill OpenCL buffers
            clCosineSimMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clCosineSimMap, repetitionMap);

            // Create kernel and set args
            kernelGetPatchCosineSim = programGetPatchCosineSim.createCLKernel("kernelGetPatchCosineSim2D");

            argn = 0;
            kernelGetPatchCosineSim.setArg(argn++, clLocalStds);
            kernelGetPatchCosineSim.setArg(argn++, clCosineSimMap);

            // Calculate absolute difference of StdDevs
            queue.putWriteBuffer(clCosineSimMap, true);
            queue.put2DRangeKernel(kernelGetPatchCosineSim, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read results back from the OpenCL device
            queue.putReadBuffer(clCosineSimMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    repetitionMap[y * w + x] = clCosineSimMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetPatchCosineSim.release();
            clCosineSimMap.release();
            programGetPatchCosineSim.release();
        }

        if (metric == metrics[2]) { // SSIM
            showStatus("Calculating SSIM...");

            // Build OpenCL program
            String programStringGetPatchSsim = getResourceAsString(BlockRedundancy2D_.class, "kernelGetPatchSsim2D.cl");
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$WIDTH$", "" + w);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$HEIGHT$", "" + h);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BW$", "" + bW);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BH$", "" + bH);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRW$", "" + bRW);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRH$", "" + bRH);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_MEAN$", "" + patchMeanFloat);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_STD$", "" + patchStdDev);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$EPSILON$", "" + EPSILON);
            programGetPatchSsim = context.createProgram(programStringGetPatchSsim).build();
            //System.out.println(programGetPatchSsim.getBuildLog()); // Print program build log to check for errors

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            clSsimMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clSsimMap, repetitionMap);

            // Create kernel and set args
            kernelGetPatchSsim = programGetPatchSsim.createCLKernel("kernelGetPatchSsim2D");

            argn = 0;
            kernelGetPatchSsim.setArg(argn++, clPatchPixels);
            kernelGetPatchSsim.setArg(argn++, clRefPixels);
            kernelGetPatchSsim.setArg(argn++, clLocalMeans);
            kernelGetPatchSsim.setArg(argn++, clLocalStds);
            kernelGetPatchSsim.setArg(argn++, clSsimMap);

            // Calculate SSIM
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clSsimMap, true);
            queue.put2DRangeKernel(kernelGetPatchSsim, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read SSIM back from the GPU
            queue.putReadBuffer(clSsimMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    repetitionMap[y * w + x] = clSsimMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetPatchSsim.release();
            clPatchPixels.release();
            clSsimMap.release();
            programGetPatchSsim.release();
        }

        if (metric == metrics[3]) { // SSIM
            showStatus("Calculating SSIM...");

            // Build OpenCL program
            String programStringGetPatchSsim = getResourceAsString(BlockRedundancy2D_.class, "kernelGetPatchSsim2D.cl");
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$WIDTH$", "" + w);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$HEIGHT$", "" + h);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_SIZE$", "" + patchSize);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BW$", "" + bW);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BH$", "" + bH);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRW$", "" + bRW);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$BRH$", "" + bRH);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_MEAN$", "" + patchMeanFloat);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$PATCH_STD$", "" + patchStdDev);
            programStringGetPatchSsim = replaceFirst(programStringGetPatchSsim, "$EPSILON$", "" + EPSILON);
            programGetPatchSsim = context.createProgram(programStringGetPatchSsim).build();
            //System.out.println(programGetPatchSsim.getBuildLog()); // Print program build log to check for errors

            // Fill OpenCL buffers
            clPatchPixels = context.createFloatBuffer(patchSize, READ_ONLY);
            fillBufferWithFloatArray(clPatchPixels, patchPixelsFloat);

            clSsimMap = context.createFloatBuffer(wh, READ_WRITE);
            fillBufferWithFloatArray(clSsimMap, repetitionMap);

            // Create kernel and set args
            kernelGetPatchSsim = programGetPatchSsim.createCLKernel("kernelGetPatchSsim2D");

            argn = 0;
            kernelGetPatchSsim.setArg(argn++, clPatchPixels);
            kernelGetPatchSsim.setArg(argn++, clRefPixels);
            kernelGetPatchSsim.setArg(argn++, clLocalMeans);
            kernelGetPatchSsim.setArg(argn++, clLocalStds);
            kernelGetPatchSsim.setArg(argn++, clSsimMap);

            // Calculate SSIM
            queue.putWriteBuffer(clPatchPixels, true);
            queue.putWriteBuffer(clSsimMap, true);
            queue.put2DRangeKernel(kernelGetPatchSsim, 0, 0, w, h, 0, 0);
            queue.finish();

            // Read SSIM back from the GPU
            queue.putReadBuffer(clSsimMap, true);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    repetitionMap[y * w + x] = clSsimMap.getBuffer().get(y * w + x);
                    queue.finish();
                }
            }
            queue.finish();

            // Release GPU resources
            kernelGetPatchSsim.release();
            clPatchPixels.release();
            clSsimMap.release();
            programGetPatchSsim.release();
        }


        // --------------------------------------- //
        // ---- Filter out irrelevant regions ---- //
        // --------------------------------------- //

        if (filterConstant > 0.0f) {
            showStatus("Calculating relevance map...");

            // Create OpenCL program
            String programStringGetRelevanceMap = getResourceAsString(BlockRedundancy2D_.class, "kernelGetRelevanceMap2D.cl");
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
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    relevanceMap[y * w + x] = clRelevanceMap.getBuffer().get(y * w + x);
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
            for (int j = bRH; j < h - bRH; j++) {
                for (int i = bRW; i < w - bRW; i++) {
                    noiseMeanVar += relevanceMap[j * w + i];
                    n += 1.0f;
                }
            }
            noiseMeanVar /= n;

            // Filter out irrelevant regions
            for (int j = bRH; j < h - bRH; j++) {
                for (int i = bRW; i < w - bRW; i++) {
                    if (relevanceMap[j * w + i] <= noiseMeanVar * filterConstant) {
                        repetitionMap[j * w + i] = 0.0f;
                    }
                }
            }

            if (normalizeOutput) {
                // ----------------------------------------------------------------------- //
                // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
                // ----------------------------------------------------------------------- //

                // Find min and max within the relevance mask
                float pearsonMin = Float.MAX_VALUE;
                float pearsonMax = -Float.MAX_VALUE;

                for (int j = bRH; j < h - bRH; j++) {
                    for (int i = bRW; i < w - bRW; i++) {
                        if (relevanceMap[j * w + i] > noiseMeanVar * filterConstant) {
                            float pixelValue = repetitionMap[j * w + i];
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
                            repetitionMap[j * w + i] = (repetitionMap[j * w + i] - pearsonMin) / (pearsonMax - pearsonMin + EPSILON);
                        }
                    }
                }
            }
        }

        // Release resources
        context.release();


        // ------------------------- //
        // ---- Display results ---- //
        // ------------------------- //

        FloatProcessor fp1 = new FloatProcessor(w, h, repetitionMap);
        ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);

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

        // Display
        imp1.show();


        // -------------------- //
        // ---- Stop timer ---- //
        // -------------------- //

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

    public static String replaceFirst(String source, String target, String replacement) {
        int index = source.indexOf(target);
        if (index == -1) {
            return source;
        }

        return source.substring(0, index)
                .concat(replacement)
                .concat(source.substring(index+target.length()));
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

