import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
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
import static java.lang.Math.min;
import static nanoj.core2.NanoJCL.replaceFirst;

public class MapNoise_ implements PlugIn {

    // OpenCL formats
    static private CLContext context;

    static private CLProgram programGetLocalStatistics, programGetPatchPearson, programGetPatchNrmse, programGetPatchSsim,
            programGetPatchHu, programGetPatchEntropy, programGetPatchPhaseCorrelation;

    static private CLKernel kernelGetLocalStatistics, kernelGetPatchPearson, kernelGetPatchNrmse, kernelGetPatchSsim,
            kernelGetPatchHu, kernelGetPatchEntropy, kernelGetPatchPhaseCorrelation;

    static private CLPlatform clPlatformMaxFlop;

    static private CLCommandQueue queue;

    private CLBuffer<FloatBuffer> clRefPatch, clRefPatchMeanSub, clRefPixels, clLocalMeans, clLocalStds, clPearsonMap,
            clNrmseMap, clMaeMap, clPsnrMap,clSsimMap, clHuMap, clEntropyMap, clPhaseCorrelationMap, clGaussianKernel;


    @Override
    public void run(String s) {

        // ---- Dialog box ----
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("Map noise");
        gd.addMessage("This Plugin predicts the positions of noise pixels. Note: the noise variance of the input " +
                "image will be stabilised using the Generalised Anscombe Transform.");
        gd.addNumericField("Box length in pixels: ", 21, 2);
        gd.showDialog();
        if (gd.wasCanceled()) return;


        // ---- Start timer ----
        long start = System.currentTimeMillis();


        // ---- Patch parameters ----
        int bW = (int) gd.getNextNumber(); // Patch width
        int bH = bW; // Patch height
        int patchSize = bW * bH;
        int bRW = bH / 2;
        int bRH = bW / 2;
        int downScale = 0; // Downscale factor
        float EPSILON = 0.0000001f;


        // ---- Initialize OpenCL ----
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


        // ---- Get reference image and some parameters ----
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

        IJ.log("Stabilising noise variance...");

        // ---- Stabilize noise variance using the Generalized Anscombe's transform ----
        // Run minimizer to find optimal gain, sigma and offset that minimize the error from a noise variance of 1
        GATMinimizer2D minimizer = new GATMinimizer2D(refPixels, w, h, 0, 100, 0);
        minimizer.run();

        // Get gain, sigma, offset and error from the minimizer and transform pixel values
        float[] pixelsGAT;
        pixelsGAT = VarianceStabilisingTransform2D_.getGAT(refPixels, minimizer.gain, minimizer.sigma, minimizer.offset);

        IJ.log("Normalising image...");

        // ---- Normalize image ----
        float[] ogMinMax = findMinMax(pixelsGAT, w, h, 0, 0);
        float ogRange = ogMinMax[1] - ogMinMax[0];
        pixelsGAT = normalize(pixelsGAT, w, h, 0, 0, ogMinMax, EPSILON);
        float[] newMinMax = findMinMax(pixelsGAT, w, h, bRW, bRH);
        float newRange = newMinMax[1] - newMinMax[0];

        IJ.log("Calculating local standard deviations...");

        // ---- Calculate mean and STD on GPU ----
        // Write image to GPU
        clRefPixels = context.createFloatBuffer(wh, READ_ONLY);
        fillBufferWithFloatArray(clRefPixels, pixelsGAT);
        queue.putWriteBuffer(clRefPixels, false);

        // Create OpenCL program
        String programStringGetLocalStatistics = getResourceAsString(RedundancyMap_.class, "kernelGetLocalMeans.cl");
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$WIDTH$", "" + w);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$HEIGHT$", "" + h);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$BW$", "" + bW);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$BH$", "" + bH);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$PATCH_SIZE$", "" + patchSize);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$BRW$", "" + bRW);
        programStringGetLocalStatistics = replaceFirst(programStringGetLocalStatistics, "$BRH$", "" + bRH);
        programGetLocalStatistics = context.createProgram(programStringGetLocalStatistics).build();

        // Create, fill and write buffers
        float[] gaussianKernel = makeGaussianKernel(bW, 0.5f);
        clGaussianKernel = context.createFloatBuffer(patchSize, READ_ONLY);
        fillBufferWithFloatArray(clGaussianKernel, gaussianKernel);
        queue.putWriteBuffer(clGaussianKernel, false);

        float[] localMeans = new float[wh];
        clLocalMeans = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalMeans, localMeans);
        queue.putWriteBuffer(clLocalMeans, false);

        float[] localStds = new float[wh];
        clLocalStds = context.createFloatBuffer(wh, READ_WRITE);
        fillBufferWithFloatArray(clLocalStds, localStds);
        queue.putWriteBuffer(clLocalStds, false);

        // Create kernel and set kernel arguments
        kernelGetLocalStatistics = programGetLocalStatistics.createCLKernel("kernelGetLocalMeans");

        int argn = 0;
        kernelGetLocalStatistics.setArg(argn++, clRefPixels);
        kernelGetLocalStatistics.setArg(argn++, clLocalMeans);
        kernelGetLocalStatistics.setArg(argn++, clLocalStds);
        kernelGetLocalStatistics.setArg(argn++, clGaussianKernel);

        // Calculate
        int nXBlocks = w/64 + ((w%64==0)?0:1);
        int nYBlocks = h/64 + ((h%64==0)?0:1);
        for(int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, h-nYB*64);
            for(int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, w-nXB*64);
                showStatus("Calculating local means... blockX="+nXB+"/"+nXBlocks+" blockY="+nYB+"/"+nYBlocks);
                queue.put2DRangeKernel(kernelGetLocalStatistics, nXB*64, nYB*64, xWorkSize, yWorkSize, 0, 0);
                queue.finish();
            }
        }

        // Read the local means map back from the GPU
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<h; y++) {
            for(int x=0; x<w; x++) {
                localMeans[y*w+x] = clLocalMeans.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Read the local stds map back from the GPU
        queue.putReadBuffer(clLocalStds, true);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                localStds[y*w+x] = clLocalStds.getBuffer().get(y*w+x);
                queue.finish();
            }
        }

        // Release resources
        kernelGetLocalStatistics.release(); // TODO: MAKES SENSE TO RELEASE KERNELS HERE? WILL THEY BE USED AGAIN?
        programGetLocalStatistics.release();

        // Create noise mask
        float error = 0; // TODO: SET TO ZERO FOR TESTING; SHOULD BE DEVIATION FROM "VARIANCE = 1"
        float[] noiseMask = new float[wh];
        float finalError = ((error*newRange)/ogRange)/2; // Normalise error to the range of the normalised image and divide by two
        IJ.log("Original range: "+ogRange);
        IJ.log("Original error: "+error);
        IJ.log("Normalized min max:"+newRange);
        IJ.log("normalized error: "+finalError);
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                if(localStds[y*w+x]<finalError){
                    noiseMask[y*w+x] = 0.0f;
                }else{
                    noiseMask[y*w+x] = pixelsGAT[y*w+x];
                }
            }
        }
        System.out.println(error);

        // ---- Display results ----
        IJ.log("Preparing results for display...");
        ImageStack ims = new ImageStack(w, h);

        FloatProcessor fp1 = new FloatProcessor(w, h, pixelsGAT);
        ims.addSlice("Variance-stabilised image", fp1);
        FloatProcessor fp2 = new FloatProcessor(w, h, localStds);
        ims.addSlice("Local Stds", fp2);
        FloatProcessor fp3 = new FloatProcessor(w, h, noiseMask);
        ims.addSlice("Noise mask", fp3);


        ImagePlus ip0 = new ImagePlus("Redundancy Maps", ims);
        ip0.show();
    }

    // ---- USER FUNCTIONS ----
    public static float[] findMinMax(float[] inputArray, int w, int h, int offsetX, int offsetY){
        float[] minMax = {inputArray[offsetY*w+offsetX], inputArray[offsetY*w+offsetX]};

        for(int j=offsetY+1; j<h-offsetY; j++){
            for(int i=offsetX+1; i<w-offsetX; i++){
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

    public static float[] normalize(float[] rawPixels, int w, int h, int offsetX, int offsetY, float[] minMax, float EPSILON){
        float range = minMax[1] - minMax[0];
        float min = minMax[0];
        float[] normalizedPixels = new float[w*h];

        for(int j=offsetY; j<h-offsetY; j++) {
            for (int i=offsetX; i<w-offsetX; i++) {
                normalizedPixels[j*w+i] = (rawPixels[j*w+i]-min)/range;
            }
        }
        return normalizedPixels;
    }

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

    public float[] makeGaussianKernel(int size, float sigma){
        float[] kernel = new float[size*size];
        float sumTotal = 0;

        int radius = size/2;
        float distance = 0;

        float euler = (float) (1.0f / (2.0 * Math.PI * (sigma*sigma)));

        for(int j=-radius; j<=radius; j++){
            for(int i=-radius; i<=radius; i++){
                distance = ((i*i)+(j*j)) / (2 * (sigma*sigma));
                kernel[(j+radius)*size+(i+radius)] = (float) (euler * Math.exp(-distance));
                sumTotal += kernel[(j+radius)*size+(i+radius)];
            }
        }

        for(int i=0; i<size*size; i++){
            kernel[i] = kernel[i] / sumTotal;
        }

        return kernel;
    }
}
