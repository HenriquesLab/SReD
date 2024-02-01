import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import java.nio.FloatBuffer;
import java.util.Arrays;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;

public class OptimiseRelevanceMap_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;
    static private CLProgram programGetLocalStatistics;
    static private CLKernel kernelGetLocalStatistics;
    static private CLPlatform clPlatformMaxFlop;
    static private CLCommandQueue queue;
    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds;

    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Display dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Optimise relevance map");
        gd.addMessage("Optimise relevance map!");
        gd.addNumericField("Block width (px): ", 3);
        gd.addNumericField("Block height (px): ", 3);
        gd.addNumericField("Filter constant step: ", 1.0f);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Retrieve dialog box parameters
        int bW = (int) gd.getNextNumber(); // Block width
        int bH = (int) gd.getNextNumber(); // Block height
        int bRW = bH / 2;
        int bRH = bW / 2;
        float step = (float) gd.getNextNumber();
        boolean useDevice = gd.getNextBoolean();


        //---------------------- //
        // ---- Start timer ---- //
        //---------------------- //

        long start = System.currentTimeMillis();


        // ---------------------------------- //
        // ---- Get image and parameters ---- //
        // ---------------------------------- //

        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = fp0.getWidth();
        int h = fp0.getHeight();
        int wh = w * h;

        // Get final block size (after removing pixels outside the inbound circle/ellipse)
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

        // --------------------------------------------------------------------------- //
        // ---- Stabilize noise variance using the Generalized Anscombe transform ---- //
        // --------------------------------------------------------------------------- //

        GATMinimizer2D minimizer = new GATMinimizer2D(refPixels, w, h, 0, 100, 0);
        minimizer.run();

        // Get gain, sigma and offset from the minimizer and transform pixel values
        refPixels = VarianceStabilisingTransform2D_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);


        // ------------------- //
        // ---- Normalize ---- //
        // ------------------- //

        // Cast to "double" type
        double[] refPixelsDouble = new double[w * h];
        for (int i = 0; i < w * h; i++) {
            refPixelsDouble[i] = (double) refPixels[i];
        }

        // Find min and max
        double imgMin = Double.MAX_VALUE;
        double imgMax = -Double.MAX_VALUE;
        for (int i = 0; i < w * h; i++) {
            double pixelValue = refPixelsDouble[i];
            imgMin = min(imgMin, pixelValue);
            imgMax = max(imgMax, pixelValue);
        }

        // Remap pixels
        for (int i = 0; i < w * h; i++) {
            refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + (double) EPSILON);
        }

        // Cast back to float
        for (int i = 0; i < w * h; i++) {
            refPixels[i] = (float) refPixelsDouble[i];
        }


        // --------------------------- //
        // ---- Initialize OpenCL ---- //
        // --------------------------- //

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

        // Create context
        context = CLContext.create(clPlatformMaxFlop);

        // Choose the best device (i.e., Filter out CPUs if GPUs are available (FLOPS calculation was giving
        // higher ratings to CPU vs. GPU))
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


        // --------------------------------------------- //
        // ---- Calculate local standard deviations ---- //
        // --------------------------------------------- //

        // Write input image to the OpenCL device
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        GlobalRedundancy.fillBufferWithFloatArray(clRefPixels, refPixels);
        queue.putWriteBuffer(clRefPixels, true);

        // Create OpenCL program
        String programStringGetLocalStatistics = GlobalRedundancy.getResourceAsString(RelevanceMap2D_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$WIDTH$", "" + w);
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$HEIGHT$", "" + h);
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$BRW$", "" + bRW);
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$BRH$", "" + bRH);
        programStringGetLocalStatistics = GlobalRedundancy.replaceFirst(programStringGetLocalStatistics, "$EPSILON$", "" + EPSILON);
        programGetLocalStatistics = context.createProgram(programStringGetLocalStatistics).build();

        // Create, fill and write buffers
        float[] localMeans = new float[wh];
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        GlobalRedundancy.fillBufferWithFloatArray(clLocalMeans, localMeans);
        queue.putWriteBuffer(clLocalMeans, true);

        float[] localStds = new float[wh];
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);
        GlobalRedundancy.fillBufferWithFloatArray(clLocalStds, localStds);
        queue.putWriteBuffer(clLocalStds, true);

        // Create kernel and set kernel arguments
        kernelGetLocalStatistics = programGetLocalStatistics.createCLKernel("kernelGetLocalMeans");

        int argn = 0;
        kernelGetLocalStatistics.setArg(argn++, clRefPixels);
        kernelGetLocalStatistics.setArg(argn++, clLocalMeans);
        kernelGetLocalStatistics.setArg(argn++, clLocalStds);

        // Calculate
        showStatus("Calculating local statistics...");
        queue.put2DRangeKernel(kernelGetLocalStatistics, 0, 0, w, h, 0, 0);
        queue.finish();

        // Read the local stds map back from the OpenCL device
        queue.putReadBuffer(clLocalStds, true);
        for (int y = bRH; y < h - bRH; y++) {
            for (int x = bRW; x < w - bRW; x++) {
                localStds[y * w + x] = clLocalStds.getBuffer().get(y * w + x);
                queue.finish();
            }
        }

        // Release resources
        context.release();
        System.out.println("context released");

        // ------------------------------------------------------------ //
        // ---- Calculate relevance maps to get optimised constant ---- //
        // ------------------------------------------------------------ //

        showStatus("Optimising relevance map...");

        float[] relevanceMapTemp = new float[wh];
        float filterConstant = 0.0f;
        float allOnes = 0.0f; //  To store the last filter constant where relevance map is all 1s
        float allZeros = 0.0f; //  To store the 1st filter constant where relevance map is all 0s

        do {
            relevanceMapTemp = getRelevanceMap(refPixels, w, h, wh, bRW, bRH, filterConstant, EPSILON, localStds);

            if (isAllOnes(relevanceMapTemp, w, h, bRW, bRH)) {
                allOnes = filterConstant;
            }

            if (isAllZeros(relevanceMapTemp)) {
                allZeros = filterConstant;
                break;
            } else {
                filterConstant += step;
            }
        } while (true);
            showStatus("Optimised!");
            IJ.log(" Filter constant range: " + allOnes + " - " + allZeros);


    }


    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //


    public static float[] getRelevanceMap(float[] refPixels, int w, int h, int wh, int bRW, int bRH, float filterConstant, float EPSILON, float[] localStds) {

    // --------------------------------- //
    // ---- Calculate Relevance Map ---- //
    // --------------------------------- //

    int blockWidth, blockHeight;
    int CIF = 352 * 288; // Resolution of a CIF file

    if (wh <= CIF) {
        blockWidth = 8;
        blockHeight = 8;
    } else {
        blockWidth = 16;
        blockHeight = 16;
    }

    int nBlocksX = w / blockWidth; // number of blocks in each row
    int nBlocksY = h / blockHeight; // number of blocks in each column
    int nBlocks = nBlocksX * nBlocksY; // total number of blocks
    float[] localVars = new float[nBlocks];
    Arrays.fill(localVars, 0.0f);

    int index = 0;

    // Calculate local variances
    for (int y = 0; y < nBlocksY; y++) {
        for (int x = 0; x < nBlocksX; x++) {
            float[] meanVar = getMeanAndVarBlock(refPixels, w, x * blockWidth, y * blockHeight, (x + 1) * blockWidth, (y + 1) * blockHeight);
            localVars[index] = (float) meanVar[1];
            index++;
        }
    }

    // Sort the local variances
    float[] sortedVars = new float[nBlocks];
    Arrays.fill(sortedVars, 0.0f);

    index = 0;
    for (int i = 0; i < nBlocks; i++) {
        sortedVars[index] = localVars[index];
        index++;
    }
    Arrays.sort(sortedVars);

    // Get the 3% lowest variances and calculate their average
    int nVars = (int) (0.03f * (float) nBlocks + 1.0f); // Number of blocks corresponding to 3% of the total amount of blocks
    float noiseVar = 0.0f;

    for (int i = 0; i < nVars; i++) {
        noiseVar += sortedVars[i];
    }
    noiseVar = abs(noiseVar / (float) nVars);
    noiseVar = (1.0f + 0.001f * (noiseVar - 40.0f)) * noiseVar;

    // Build the relevance map
    float[] relevanceMap = new float[wh];
    Arrays.fill(relevanceMap, 0.0f);

    float threshold;
    if (noiseVar == 0.0f) {
        //IJ.log("WARNING: Noise variance is 0. Adjust the relevance threshold using the filter constant directly.");
        threshold = filterConstant;
        //IJ.log("Threshold: " + filterConstant);
    } else {
        //IJ.log("Noise variance: " + noiseVar);
        threshold = noiseVar * filterConstant;
        //IJ.log("Relevance threshold: " + threshold);
    }

    double nPixels = 0.0; // Number of relevant pixels
    for (int j = bRH; j < h - bRH; j++) {
        for (int i = bRW; i < w - bRW; i++) {
            float var = localStds[j * w + i] * localStds[j * w + i];
            if (var < threshold || var == 0.0f) {
                relevanceMap[j * w + i] = 0.0f;
            } else {
                relevanceMap[j * w + i] = 1.0f;
                nPixels += 1.0;
            }
        }
    }
        return relevanceMap;
    }

    // Get mean and variance of a patch
    public static float[] getMeanAndVarBlock(float[] pixels, int width, int xStart, int yStart, int xEnd, int yEnd) {
        float mean = 0;
        float var;

        float sq_sum = 0;

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

        return new float[] {mean, var};
    }

    public boolean isAllZeros(float[] array) {
        for (float f : array) {
            if (f != 0.0f) {
                return false;
            }
        }
        return true;
    }

    public boolean isAllOnes(float[] imageArray, int w, int h, int bRW, int bRH) {
        for (int j=bRH; j<h-bRH; j++) {
            for (int i=bRW; i<w-bRW; i++) {
                if (imageArray[j*w+i] != 1.0f) {
                    return false;
                }
            }
        }
        return true;
    }
}

