import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.ImageCalculator;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.nio.FloatBuffer;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static java.lang.Math.*;

public class RelevanceMaskStack2D_ implements PlugIn {

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
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Generate Relevance Mask stack");
        gd.addNumericField("Block width (px): ", 3);
        gd.addNumericField("Block height (px): ", 3);
        gd.addNumericField("Filter constant lower limit: ", 0.0f);
        gd.addNumericField("Filter constant upper limit: ", 1.0f);
        gd.addNumericField("Filter constant step: ", 0.1f);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Retrieve dialog box parameters
        int bW = (int) gd.getNextNumber(); // Block width
        int bH = (int) gd.getNextNumber(); // Block height
        int bRW = bH / 2;
        int bRH = bW / 2;
        float lowerLimit = (float) gd.getNextNumber();
        float upperLimit = (float) gd.getNextNumber();
        float step = (float) gd.getNextNumber();
        boolean useDevice = gd.getNextBoolean();


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


        // ---------------------------------- //
        // ---- Calculate relevance maps ---- //
        // ---------------------------------- //

        showStatus("Calculating relevance maps...");
        IJ.log("Calculating relevance maps...");

        float[] relevanceMapTemp = new float[wh];
        float filterConstant = lowerLimit;

        int nIter = (int)Math.ceil((upperLimit-lowerLimit)/step)+1;
        ImageStack imsOutput = new ImageStack(w, h, nIter);
        ImageStack imsInput = new ImageStack(w, h, nIter);

        FloatProcessor fpInput = new FloatProcessor(w, h, refPixels);

        for(int i=0;i<nIter;i++){
            relevanceMapTemp = OptimiseRelevanceMap_.getRelevanceMap(refPixels, w, h, wh, bRW, bRH, filterConstant, EPSILON, localStds);
            FloatProcessor ipOutput = new FloatProcessor(w, h, relevanceMapTemp);
            imsOutput.setProcessor(ipOutput, i+1);
            imsOutput.setSliceLabel(String.valueOf(filterConstant), i+1);

            imsInput.setProcessor(fpInput, i+1);

            filterConstant += step;
        }

        ImagePlus impInput = new ImagePlus("Input", imsInput);
        ImagePlus impOutput = new ImagePlus("Output", imsOutput);

        impInput.show();
        IJ.run(impInput, "Grays", "");

        impOutput.show();
        IJ.run(impOutput, "Magenta", "");

        IJ.run(impInput, "Merge Channels...", "c1=Input c2=Output create");

        IJ.log("Done!");

    }


    // ------------------------ //
    // ---- USER FUNCTIONS ---- //
    // ------------------------ //


}

