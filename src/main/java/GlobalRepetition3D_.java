import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.measure.Calibration;
import ij.plugin.LutLoader;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.LUT;

import java.awt.image.IndexColorModel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;

public class GlobalRepetition3D_ implements PlugIn {
    static private CLContext context;
    static private CLPlatform clPlatformMaxFlop;
    static private CLCommandQueue queue;
    static private CLProgram programGetLocalMeans3D, programGetCosineSimMap3D, programGetRelevanceMap3D;

    static private CLKernel kernelGetLocalMeans3D, kernelGetCosineSimMap3D, kernelGetRelevanceMap3D;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds, clCosineSimMap, clRelevanceMap, clWeightsSumMap;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Define metric possibilities
        String[] metrics = new String[2];
        metrics[0] = "Pearson's R";
        metrics[1] = "Cosine similarity";

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Global Repetition (3D)");
        gd.addNumericField("Block width (px): ", 3, 2);
        gd.addNumericField("Block height (px): ", 3, 2);
        gd.addNumericField("Block depth (px): ", 3, 2);
        gd.addSlider("Filter constant: ", 0.0f, 10.0f, 0.0, 0.1f);
        gd.addChoice("Metric:", metrics, metrics[1]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if(gd.wasCanceled()) return;

        // Get dialog parameters
        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = (int) gd.getNextNumber(); // Patch height
        int bZ = (int) gd.getNextNumber(); // Patch depth

        float filterConstant = (float) gd.getNextNumber();

        String metric = gd.getNextChoice();

        boolean normalizeOutput = gd.getNextBoolean();

        boolean useDevice = gd.getNextBoolean();

        // Check if patch dimensions are odd, otherwise kill program
        if(bW%2==0 || bH%2==0) {
            IJ.error("Block dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        // Check if patch has at least 3 slices, otherwise kill program
        if(bZ<3) {
            IJ.error("Block must have at least 3 slices. Please try again.");
            return;
        }

        // Calculate block radius
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)
        int bRZ = bZ/2; // Patch radius (z-axis)

        // Get ImagePlus and ImageStack
        ImagePlus imp0 = WindowManager.getCurrentImage();
        if(imp0==null) {
            IJ.error("Image not found. Try again.");
            return;
        }

        ImageStack ims0 = imp0.getStack();

        // Get calibration parameters
        Calibration calibration = imp0.getCalibration();

        // Get image dimensions
        int w = ims0.getWidth();
        int h = ims0.getHeight();
        int z = ims0.getSize();
        int wh = w*h;
        int whz = w*h*z;


        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        IJ.log("SReD has started, please wait.");
        long start = System.currentTimeMillis();


        // ---------------------------------- //
        // ---- Stabilize noise variance ---- //
        // ---------------------------------- //

        IJ.log("Stabilizing noise variance of the image...");

        float[][] refPixels = new float[z][wh];
        for(int n=0; n<z; n++) {
            for(int y=0; y<h; y++) {
                for(int x=0; x<w; x++) {
                    refPixels[n][y*w+x] = ims0.getProcessor(n+1).convertToFloatProcessor().getf(x, y);
                }
            }
        }

        GATMinimizer3D minimizer = new GATMinimizer3D(refPixels, w, h, z, 0, 100, 0);
        minimizer.run();

        for(int n=0; n<z; n++) {
            refPixels[n] = VarianceStabilisingTransform3D_.getGAT(refPixels[n], minimizer.gain, minimizer.sigma, minimizer.offset);
        }


        // ----------------------- //
        // ---- Process image ---- //
        // ----------------------- //

        // Cast to double type and store as flattened 1D array
        double[] refPixelsDouble = new double[whz];
        for(int n=0; n<z; n++) {
            for(int y=0; y<h; y++) {
                for(int x=0; x<w; x++) {
                    refPixelsDouble[w*h*n+y*w+x] = refPixels[n][y*w+x];
                }
            }
        }

        // Get min and max
        double imgMin = Double.MAX_VALUE; // Initialize as a very large number
        double imgMax = -Double.MAX_VALUE; // Initialize as a very small number
        for(int i=0; i<whz; i++) {
            double pixelValue = refPixelsDouble[i];
            if(pixelValue<imgMin) {
                imgMin = pixelValue;
            }
            if(pixelValue>imgMax) {
                imgMax = pixelValue;
            }
        }

        // Normalize
        for(int i=0; i<whz; i++) {
            refPixelsDouble[i] = (refPixelsDouble[i]-imgMin)/(imgMax-imgMin+EPSILON);
        }

        // Cast back to float
        float[] refPixelsFloat = new float[whz];
        for(int i=0; i<whz; i++) {
            refPixelsFloat[i] = (float)refPixelsDouble[i];
        }

        // Get final patch size (after removing pixels outside the sphere/ellipsoid)
        int patchSize = 0;
        for(int n=0; n<bZ; n++) {
            for(int y=0; y<bH; y++) {
                for(int x=0; x<bW; x++) {
                    float dx = (float) (x-bRW);
                    float dy = (float) (y-bRH);
                    float dz = (float) (n-bRZ);
                    if(((dx*dx)/(float)(bRW*bRW)) + ((dy*dy)/(float)(bRH*bRH)) + ((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f) {
                        patchSize++;
                    }
                }
            }
        }

        // --------------------------- //
        // ---- Initialize OpenCL ---- //
        // --------------------------- //

        // Check OpenCL devices
        CLPlatform[] allPlatforms = CLPlatform.listCLPlatforms();

        try{
            allPlatforms = CLPlatform.listCLPlatforms();
        }catch (CLException ex) {
            IJ.log("Something went wrong while initialising OpenCL.");
            throw new RuntimeException("Something went wrong while initialising OpenCL.");
        }

        double nFlops = 0;

        for(CLPlatform allPlatform : allPlatforms) {
            CLDevice[] allCLdeviceOnThisPlatform = allPlatform.listCLDevices();

            for(CLDevice clDevice : allCLdeviceOnThisPlatform) {
                IJ.log("--------");
                IJ.log("Device name: " + clDevice.getName());
                IJ.log("Device type: " + clDevice.getType());
                IJ.log("Max clock: " + clDevice.getMaxClockFrequency() + " MHz");
                IJ.log("Number of compute units: " + clDevice.getMaxComputeUnits());
                IJ.log("Max work group size: " + clDevice.getMaxWorkGroupSize());
                if(clDevice.getMaxComputeUnits() * clDevice.getMaxClockFrequency() > nFlops) {
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
        for(int i=0; i<allDevices.length; i++){
            if(allDevices[i].getType() == CLDevice.Type.GPU){
                hasGPU = true;
            }
        }
        CLDevice chosenDevice;
        if(hasGPU){
            chosenDevice = context.getMaxFlopsDevice(CLDevice.Type.GPU);
        }else{
            chosenDevice = context.getMaxFlopsDevice();
        }

        // Get chosen device from preferences
        if(useDevice){
            String deviceName = Prefs.get("SReD.OpenCL.device", null);
            for(CLDevice device:allDevices){
                if(device.getName().equals(deviceName)){
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

        IJ.log("Calculating noise variance...");
        IJ.showStatus("Calculating noise variance...");


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(whz, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(whz, READ_WRITE);
        clLocalStds = context.createFloatBuffer(whz, READ_WRITE);

        // Create OpenCL program
        String programStringGetLocalMeans3D = getResourceAsString(BlockRedundancy3D_.class, "kernelGetPatchMeans3D.cl");
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$WIDTH$", "" + w);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$HEIGHT$", "" + h);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$DEPTH$", "" + z);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$BRW$", "" + bRW);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$BRH$", "" + bRH);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$BRZ$", "" + bRZ);
        programStringGetLocalMeans3D = replaceFirst(programStringGetLocalMeans3D, "$EPSILON$", "" + EPSILON);
        programGetLocalMeans3D = context.createProgram(programStringGetLocalMeans3D).build();

        // Fill OpenCL buffers
        fillBufferWithFloatArray(clRefPixels, refPixelsFloat);

        float[] localMeans = new float[whz];
        fillBufferWithFloatArray(clLocalMeans, localMeans);

        float[] localStds = new float[whz];
        fillBufferWithFloatArray(clLocalStds, localStds);

        // Create OpenCL kernel and set arguments
        kernelGetLocalMeans3D = programGetLocalMeans3D.createCLKernel("kernelGetPatchMeans3D");

        int argn = 0;
        kernelGetLocalMeans3D.setArg(argn++, clRefPixels);
        kernelGetLocalMeans3D.setArg(argn++, clLocalMeans);
        kernelGetLocalMeans3D.setArg(argn++, clLocalStds);

        // Write buffers
        queue.putWriteBuffer(clRefPixels, true);
        queue.putWriteBuffer(clLocalMeans, true);
        queue.putWriteBuffer(clLocalStds, true);

        // Calculate
        showStatus("Calculating local statistics...");
        //queue.put3DRangeKernel(kernelGetLocalMeans3D, 0, 0, 0, w, h, z, 0, 0, 0);
        //queue.finish();

        // Calculate Pearson's correlation coefficient
        int nXBlocks = w/64 + ((w%64==0)?0:1);
        int nYBlocks = h/64 + ((h%64==0)?0:1);
        //int nZBlocks = z/64 + ((z%64==0)?0:1);
        int nZBlocks = z/bZ + ((z%bZ==0)?0:1); // This tries to reduce workload by creating blocks with the minimum z-size required for calculations
        float totalBlocks = nXBlocks*nYBlocks*nZBlocks;
        float currentBlock = 1.0f;

        for(int nZB=0; nZB<nZBlocks; nZB++){
            int zWorkSize = min(bZ, z-nZB*bZ);
            for(int nYB=0; nYB<nYBlocks; nYB++){
                int yWorkSize = min(64, h-nYB*64);
                for(int nXB=0; nXB<nXBlocks; nXB++){
                    int xWorkSize = min(64, w-nXB*64);
                    float progress = (currentBlock/totalBlocks)*100.0f;
                    showStatus("Calculating local statistics... " + (int)progress + "%");
                    queue.put3DRangeKernel(kernelGetLocalMeans3D, nXB*64, nYB*64, nZB*bZ, xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    queue.finish();
                    currentBlock += 1.0f;
                }
            }
        }

        // Read the local means map back from the device
        queue.putReadBuffer(clLocalMeans, true);
        for(int n=0; n<z; n++){
            for(int y=0; y<h; y++){
                for(int x=0; x<w; x++){
                    int index = w*h*n+y*w+x;
                    localMeans[index] = clLocalMeans.getBuffer().get(index);
                }
            }
        }
        queue.finish();

        // Read the local stds map back from the device
        queue.putReadBuffer(clLocalStds, true);
        for(int n=0; n<z; n++){
            for(int y=0; y<h; y++){
                for(int x=0; x<w; x++){
                    int index = w*h*n+y*w+x;
                    localStds[index] = clLocalStds.getBuffer().get(index);
                }
            }
        }
        queue.finish();

        // Release memory
        kernelGetLocalMeans3D.release();
        programGetLocalMeans3D.release();

        // TODO: THIS IS FOR TESTING ONLY
        ImageStack imsFinal1 = new ImageStack(w, h, z);
        for (int n=0; n<z; n++) {
            FloatProcessor temp = new FloatProcessor(w, h);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    temp.setf(x, y, localMeans[w*h*n+y*w+x]);
                }
            }
            imsFinal1.setProcessor(temp, n+1);
        }
        ImagePlus impFinal1 = new ImagePlus("Local means", imsFinal1);
        impFinal1.setCalibration(calibration);
        impFinal1.show();

        // TODO: THIS IS FOR TESTING ONLY
        ImageStack imsFinal2 = new ImageStack(w, h, z);
        for (int n=0; n<z; n++) {
            FloatProcessor temp = new FloatProcessor(w, h);
            for (int y=0; y<h; y++) {
                for (int x=0; x<w; x++) {
                    temp.setf(x, y, localStds[w*h*n+y*w+x]);
                }
            }
            imsFinal2.setProcessor(temp, n+1);
        }
        ImagePlus impFinal2 = new ImagePlus("local stds", imsFinal2);
        impFinal2.setCalibration(calibration);
        impFinal2.show();


        // --------------------------------- //
        // ---- Calculate relevance map ---- //
        // --------------------------------- //

        showStatus("Calculating relevance map...");

        // Define block dimensions for variance calculation
        int blockWidth, blockHeight, blockDepth;
        int CIF = 352*288;

        if(wh<=CIF){
            blockWidth = 8;
            blockHeight = 8;
            blockDepth = 8;
        }else{
            blockWidth = 16;
            blockHeight = 16;
            blockDepth = bZ; // TODO: CHANGE THIS TO LIKE 1 otherwise image would need at least 16 slices
        }

        int nBlocksX = w / blockWidth; // number of blocks in each row
        int nBlocksY = h / blockHeight; // number of blocks in each column
        int nBlocksZ = z / blockDepth; // number of blocks in depth

        int nBlocks = nBlocksX * nBlocksY * nBlocksZ; // total number of blocks
        float[] localVars = new float[nBlocks];
        Arrays.fill(localVars, 0.0f);

        int index = 0;
        for(int n=0; n<nBlocksZ; n++){
            for (int y = 0; y < nBlocksY; y++) {
                for (int x = 0; x < nBlocksX; x++) {
                    double[] meanVar = getMeanAndVarBlock3D(refPixelsFloat, w, h, z, x * blockWidth,
                            y * blockHeight, n * blockDepth, (x + 1) * blockWidth, (y + 1) * blockHeight,
                            (n + 1) * blockDepth);
                    localVars[index] = (float) meanVar[1];
                    index++;
                }
            }
        }

        // Sort the local variances
        float[] sortedVars = new float[nBlocks];
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
        float[] relevanceMap = new float[whz];
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

        float nPixels = 0.0f; // Number of relevant pixels
        for(int n=bRZ; n<z-bRZ; n++){
            for (int j = bRH; j < h - bRH; j++) {
                for (int i = bRW; i < w - bRW; i++) {
                    index = w*h*n+j*w+i;
                    float var = localStds[index] * localStds[index];
                    if (var < threshold || var == 0.0f) {
                        relevanceMap[index] = 0.0f;
                    } else {
                        relevanceMap[index] = 1.0f;
                        nPixels += 1.0;
                    }
                }
            }
        }




        // --------------------------------------------------------------- //
        // ---- Calculate block repetition map with the chosen metric ---- //
        // --------------------------------------------------------------- //

        float[] repetitionMap = new float[whz];
        Arrays.fill(repetitionMap, 0.0f);

        float[] weightSumMap = new float[whz];
        Arrays.fill(weightSumMap, 0.0f);

        if(metric==metrics[1]) { // Cosine similarity
            showStatus("Calculating global repetition... 0%");

            // Build OpenCL program
            String programStringGetCosineSim3D = getResourceAsString(GlobalRepetition3D_.class, "kernelGetCosineSimMap3D.cl");
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$WIDTH$", "" + w);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$HEIGHT$", "" + h);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$DEPTH$", "" + z);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$PATCH_SIZE$", "" + patchSize);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$BRW$", "" + bRW);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$BRH$", "" + bRH);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$BRZ$", "" + bRZ);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$FILTERPARAM$", "" + noiseVar);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$THRESHOLD$", "" + threshold);
            programStringGetCosineSim3D = replaceFirst(programStringGetCosineSim3D, "$EPSILON$", "" + EPSILON);
            programGetCosineSimMap3D = context.createProgram(programStringGetCosineSim3D).build();

            // Fill OpenCL buffers
            clCosineSimMap = context.createFloatBuffer(whz, READ_WRITE);
            fillBufferWithFloatArray(clCosineSimMap, repetitionMap);
            queue.putWriteBuffer(clCosineSimMap, true);

            clWeightsSumMap = context.createFloatBuffer(whz, READ_WRITE);
            fillBufferWithFloatArray(clWeightsSumMap, weightSumMap);
            queue.putWriteBuffer(clWeightsSumMap, true);

            // Create kernel and set args
            kernelGetCosineSimMap3D = programGetCosineSimMap3D.createCLKernel("kernelGetCosineSimMap3D");

            argn = 0;
            kernelGetCosineSimMap3D.setArg(argn++, clRefPixels);
            kernelGetCosineSimMap3D.setArg(argn++, clLocalStds);
            kernelGetCosineSimMap3D.setArg(argn++, clWeightsSumMap);
            kernelGetCosineSimMap3D.setArg(argn++, clCosineSimMap);

            // Calculate Pearson's correlation coefficient
            currentBlock = 1.0f; // Restart the counter
            for(int nZB=0; nZB<nZBlocks; nZB++){
                int zWorkSize = min(bZ, z-nZB*bZ);
                for(int nYB=0; nYB<nYBlocks; nYB++){
                    int yWorkSize = min(64, h-nYB*64);
                    for(int nXB=0; nXB<nXBlocks; nXB++){
                        int xWorkSize = min(64, w-nXB*64);
                        float progress = (currentBlock/totalBlocks)*100.0f;
                        showStatus("Calculating global repetition... " + (int)progress + "%");
                        queue.put3DRangeKernel(kernelGetCosineSimMap3D, nXB*64, nYB*64, nZB*bZ, xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                        queue.finish();
                        currentBlock += 1.0f;
                    }
                }
            }

            // Read Cosine Similarity map back from the GPU
            queue.putReadBuffer(clCosineSimMap, true);
            queue.putReadBuffer(clWeightsSumMap, true);
            for(int n=0; n<z; n++){
                for(int y=0; y<h; y++){
                    for(int x=0; x<w; x++){
                        index = w*h*n+y*w+x;

                        float similarity = clCosineSimMap.getBuffer().get(index);
                        queue.finish();

                        float weightSum = clWeightsSumMap.getBuffer().get(index);
                        queue.finish();

                        repetitionMap[index] = similarity/(weightSum*nPixels+EPSILON);
                        //System.out.println(similarity);
                    }
                }
            }

            // Release GPU resources
            kernelGetCosineSimMap3D.release();
            clCosineSimMap.release();
            clWeightsSumMap.release();
            programGetCosineSimMap3D.release();


            // ----------------------------------------------------------------------- //
            // ---- Normalize output (avoiding pixels outside the relevance mask) ---- //
            // ----------------------------------------------------------------------- //

            if(normalizeOutput){

                // Find min and max within the relevance mask
                float outputMin = Float.MAX_VALUE;
                float outputMax = -Float.MAX_VALUE;

                for(int n=bRZ; n<z-bRZ; n++){
                    for(int j=bRH; j<h-bRH; j++){
                        for(int i=bRW; i<w-bRW; i++){
                            index = w*h*n+j*w+i;
                            if(relevanceMap[index]==1.0f){
                                float pixelValue = repetitionMap[index];
                                outputMax = max(outputMax, pixelValue);
                                outputMin = min(outputMin, pixelValue);
                            }
                        }
                    }
                }

                // Remap pixels
                for(int n=bRZ; n<z-bRZ; n++){
                    for(int j=bRH; j<h-bRH; j++){
                        for(int i=bRW; i<w-bRW; i++){
                            index = w*h*n+j*w+i;
                            if(relevanceMap[index]==1.0f){
                                repetitionMap[index] = (repetitionMap[index]-outputMin)/(outputMax-outputMin);
                            }else{
                                repetitionMap[index] = 0.0f;
                            }
                        }
                    }
                }
            }


            // ------------------------- //
            // ---- Display results ---- //
            // ------------------------- //

            ImageStack imsFinal = new ImageStack(w, h, z);
            for (int n=0; n<z; n++) {
                FloatProcessor temp = new FloatProcessor(w, h);
                for (int y=0; y<h; y++) {
                    for (int x=0; x<w; x++) {
                        temp.setf(x, y, repetitionMap[w*h*n+y*w+x]);
                    }
                }
                imsFinal.setProcessor(temp, n+1);
            }
            ImagePlus impFinal = new ImagePlus("Global Repetition Map", imsFinal);
            impFinal.setCalibration(calibration);

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
                    impFinal.getProcessor().setLut(lut);
                    //imp1.updateAndDraw();
                } catch (IOException e) {
                    IJ.error("Could not load SReD LUT");
                }
            }

            impFinal.show();
        }


        // ---- Stop timer ----
        IJ.log("Finished!");
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Elapsed time: " + elapsedTime / 1000 + " sec");
        IJ.log("--------");


    }


    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for (int n = 0; n < pixels.length; n++) {
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
        while ((length = inputStream.read(buffer)) != -1) {
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
                .concat(source.substring(index + target.length()));
    }

    // Get mean and variance of a patch
    public double[] getMeanAndVarBlock3D(float[] pixels, int width, int height, int depth, int xStart, int yStart, int zStart, int xEnd, int yEnd, int zEnd) {
        double mean = 0;
        double var;

        double sq_sum = 0;

        int bWidth = xEnd-xStart;
        int bHeight = yEnd - yStart;
        int bDepth = zEnd - zStart;
        int bWHZ = bWidth*bHeight*bDepth;

        for(int z=zStart; z<zEnd; z++){
            for (int y=yStart; y<yEnd; y++) {
                for (int x=xStart; x<xEnd; x++) {
                    float v = pixels[width*height*z+y*width+x];
                    mean += v;
                    sq_sum += v * v;
                }
            }
        }
        mean = mean / bWHZ;
        var = sq_sum / bWHZ - mean * mean;

        return new double[] {mean, var};
    }

}








