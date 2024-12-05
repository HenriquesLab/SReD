import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import ij.process.FloatProcessor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static ij.IJ.showStatus;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;

public class CalculateLocalMeans_ implements PlugIn {

    static private CLContext context;

    static private CLProgram programGetPatchMeans;

    static private CLKernel kernelGetPatchMeans;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPixels, clLocalMeans, clLocalStds;


    @Override
    public void run(String s) {

        float EPSILON = 0.0000001f;


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Get all open image titles
        int  nImages = getImageCount();
        if (nImages < 1) {
            IJ.error("No images found.");
            return;
        }

        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for(int i=0; i<nImages; i++){
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Image statistics");
        gd.addChoice("Image:", titles, titles[0]);
        gd.addNumericField("Block width (px): ", 3, 2);
        gd.addNumericField("Block height (px): ", 3, 2);
        gd.addSlider("Filter constant: ", 0.0f, 5.0f, 1.0f, 0.1f);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);

        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameters from dialog box
        String imgTitle = gd.getNextChoice();
        int imgID = 0;
        for(int i=0; i<nImages; i++){
            if(titles[i].equals(imgTitle)){ // .equals() instead of "==" required to run from macro
                imgID = ids[i];
            }
        }

        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = (int) gd.getNextNumber(); // Patch height

        float filterConstant = (float) gd.getNextNumber();

        boolean normalizeOutput = gd.getNextBoolean();

        boolean useDevice = gd.getNextBoolean();


        // --------------------- //
        // ---- Start timer ---- //
        // --------------------- //

        IJ.log("SReD has started, please wait.");
        long start = System.currentTimeMillis();


        // Check if patch dimensions are odd, otherwise kill program
        if (bW % 2 == 0 || bH % 2 == 0) {
            IJ.error("Patch dimensions must be odd (e.g., 3x3 or 5x5). Please try again.");
            return;
        }

        // Calculate block radius
        int bRW = bW/2; // Patch radius (x-axis)
        int bRH = bH/2; // Patch radius (y-axis)

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


        // ------------------------------------------------- //
        // ---- Get reference image and some parameters ---- //
        // ------------------------------------------------- //

        ImagePlus imp0 = WindowManager.getCurrentImage();
        if (imp0 == null) {
            IJ.error("No image found. Please open an image and try again.");
            return;
        }

        FloatProcessor fp0 = imp0.getProcessor().convertToFloatProcessor();
        float[] refPixels = (float[]) fp0.getPixels();
        int w = fp0.getWidth();
        int h = fp0.getHeight();
        int wh = w*h;

        // --------------------------------------------------------------------------- //
        // ---- Stabilize noise variance using the Generalized Anscombe transform ---- //
        // --------------------------------------------------------------------------- //
        GATMinimizer2D minimizer = new GATMinimizer2D(refPixels, w, h, 0, 100, 0);
        minimizer.run();

        refPixels = VarianceStabilisingTransform2D_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);


        // ------------------------ //
        // ---- Normalize image---- //
        // ------------------------ //

        // Cast to "double" type
        double[] refPixelsDouble = new double[w*h];
        for(int i=0; i<wh; i++){
            refPixelsDouble[i] = (double)refPixels[i];
        }

        // Find min and max
        double imgMin = Double.MAX_VALUE;
        double imgMax = -Double.MAX_VALUE;
        for(int i=0; i<w*h; i++){
            double pixelValue = refPixelsDouble[i];
            if(pixelValue<imgMin){
                imgMin = pixelValue;
            }
            if(pixelValue>imgMax){
                imgMax = pixelValue;
            }
        }

        // Remap pixels
        for(int i=0; i<wh; i++) {
            refPixelsDouble[i] = (refPixelsDouble[i] - imgMin) / (imgMax - imgMin + (double)EPSILON);
        }

        // Cast back to float
        for(int i=0; i<wh; i++){
            refPixels[i] = (float)refPixelsDouble[i];
        }


        // ------------------------------------ //
        // ---- Calculate image statistics ---- //
        // ------------------------------------ //


        // ------------------------------- //
        // ---- Calculate local means ---- //
        // ------------------------------- //

        // Create buffers
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);

        // Create OpenCL program
        String programStringGetPatchMeans = getResourceAsString(BlockRepetition2D_.class, "kernelGetLocalStatistics2D.cl");
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
                queue.finish();

            }
        }

        // Read the local stds map back from the device
        queue.putReadBuffer(clLocalStds, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                localStds[y*w+x] = clLocalStds.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetPatchMeans.release();
        programGetPatchMeans.release();


        // ------------------------ //
        // ---- Display output ---- //
        // ------------------------ //

        ImageStack stack = new ImageStack(w, h, 2);

        FloatProcessor fp1 = new FloatProcessor(w, h, localMeans);
        FloatProcessor fp2 = new FloatProcessor(w, h, localStds);
        //ImagePlus imp1 = new ImagePlus("Block Redundancy Map", fp1);
        //imp1.show();

        stack.setProcessor(fp1, 1);
        stack.setProcessor(fp2, 2);

        ImagePlus imp = new ImagePlus("Image statistics", stack);
        imp.show();

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

    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] pixels) {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<pixels.length; n++) {
            buffer.put(n, pixels[n]);
        }
    }

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

}

