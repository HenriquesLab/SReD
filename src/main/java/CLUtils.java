import com.jogamp.opencl.*;
import ij.IJ;
import ij.Prefs;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.logging.Logger;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static ij.IJ.showStatus;
import static java.lang.Math.max;
import static java.lang.Math.min;

public class CLUtils {

    // Private constructor to prevent instantiation
    private CLUtils() {
        throw new UnsupportedOperationException("Utility class");
    }


    // ---------------------------- //
    // ---- OBJECTS FOR OPENCL ---- //
    // ---------------------------- //

    /**
     * This class encapsulates the OpenCL resources required for executing
     * OpenCL kernels, including the context, device, and command queue.
     */
    public static class OpenCLResources
    {

        private final CLContext context;
        private final CLDevice device;
        private final CLCommandQueue queue;

        /**
         * Constructs an instance of OpenCLResources with the specified OpenCL context, device, and command queue.
         *
         * @param context the OpenCL context
         * @param device  the OpenCL device
         * @param queue   the command queue associated with the OpenCL device
         */
        public OpenCLResources(CLContext context, CLDevice device, CLCommandQueue queue) {
            this.context = context;
            this.device = device;
            this.queue = queue;
        }

        /**
         * Returns the OpenCL context associated with this instance.
         *
         * @return the OpenCL context
         */
        public CLContext getContext() {
            return context;
        }

        /**
         * Returns the OpenCL device associated with this instance.
         *
         * @return the OpenCL device
         */
        public CLDevice getDevice() {
            return device;
        }

        /**
         * Returns the command queue associated with the OpenCL device.
         *
         * @return the command queue
         */
        public CLCommandQueue getQueue() {
            return queue;
        }
    }


    /**
     * This class encapsulates local statistics for a 2D image, including the local means, local standard deviations,
     * and associated OpenCL buffers.
     */
    public static class CLLocalStatistics
    {

        private float[] localMeans;
        private float[] localStds;
        private CLBuffer<FloatBuffer> clImageArray;
        private CLBuffer<FloatBuffer> clLocalMeans;
        private CLBuffer<FloatBuffer> clLocalStds;

        /**
         * Constructs an instance of CLLocalStatistics with the specified local means, local standard deviations,
         * and OpenCL buffers for the image array, local means, and local standard deviations.
         *
         * @param localMeans   an array of local means calculated from the image
         * @param localStds    an array of local standard deviations calculated from the image
         * @param clImageArray OpenCL buffer containing the image data
         * @param clLocalMeans OpenCL buffer containing the local means
         * @param clLocalStds  OpenCL buffer containing the local standard deviations
         */
        public CLLocalStatistics(float[] localMeans, float[] localStds, CLBuffer<FloatBuffer> clImageArray, CLBuffer<FloatBuffer> clLocalMeans, CLBuffer<FloatBuffer> clLocalStds) {
            this.localMeans = localMeans;
            this.localStds = localStds;
            this.clImageArray = clImageArray;
            this.clLocalMeans = clLocalMeans;
            this.clLocalStds = clLocalStds;
        }

        /**
         * Returns the local means calculated for the image.
         *
         * @return an array of local means
         */
        public float[] getLocalMeans() {
            return localMeans;
        }

        /**
         * Returns the local standard deviations calculated for the image.
         *
         * @return an array of local standard deviations
         */
        public float[] getLocalStds() {
            return localStds;
        }

        /**
         * Returns the OpenCL buffer containing the image data.
         *
         * @return the OpenCL buffer for the image array
         */
        public CLBuffer<FloatBuffer> getCLImageArray() {
            return clImageArray;
        }
        /**
         * Returns the OpenCL buffer for storing the local means.
         *
         * @return the OpenCL buffer for local means
         */
        public CLBuffer<FloatBuffer> getCLLocalMeans() {
            return clLocalMeans;
        }

        /**
         * Returns the OpenCL buffer for storing the local standard deviations.
         *
         * @return the OpenCL buffer for local standard deviations
         */
        public CLBuffer<FloatBuffer> getCLLocalStds() {
            return clLocalStds;
        }
    }


    // ------------------------------------------------------------------ //
    // ---- METHODS FOR OPENCL INITIALISATION AND RESOURCE MANAGEMENT---- //
    // ------------------------------------------------------------------ //
    /**
     * Initializes OpenCL and retrieves objects related to OpenCL resources.
     *
     * @param useDevice A boolean indicating if the OpenCL device should be chosen
     *                  according to the user-defined preference.
     * @return A {@link OpenCLResources} object containing the initialized OpenCL
     *         {@code CLContext}, {@code CLDevice}, and {@code CLCommandQueue}.
     * @throws RuntimeException if there is an error during OpenCL initialization.
     */
    public static OpenCLResources getOpenCLResources(boolean useDevice)
    {
        IJ.log("Initialising OpenCL...");

        // Check OpenCL devices
        CLPlatform[] allPlatforms = CLPlatform.listCLPlatforms();
        try {
            allPlatforms = CLPlatform.listCLPlatforms();
        } catch (CLException ex) {
            IJ.log("Something went wrong while initialising OpenCL.");
            throw new RuntimeException("Something went wrong while initialising OpenCL.");
        }

        double nFlops = 0;

        CLPlatform clPlatformMaxFlop = null;
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

        // Create OpenCL context
        CLContext context = CLContext.create(clPlatformMaxFlop);

        // Choose the best device (filter out CPUs if GPUs are available)
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

        // Get chosen device from preferences (optional)
        if (useDevice) {
            String deviceName = Prefs.get("SReD.OpenCL.device", null);
            for (CLDevice device : allDevices) {
                if (device.getName().equals(deviceName)) {
                    chosenDevice = device;
                    break;
                }
            }
        }

        IJ.log("Device chosen: " + chosenDevice.getName());

        // Create command queue
        CLCommandQueue queue = chosenDevice.createCommandQueue();

        return new OpenCLResources(context, chosenDevice, queue);
    }


    /**
     * Creates an OpenCL buffer and fills it with the provided float array.
     *
     * @param context the OpenCL context in which to create the buffer
     * @param size    the size of the buffer to be created
     * @param flag    the memory flag to specify the allocation behavior of the buffer
     * @param array   the float array containing the data to fill the buffer with
     * @return a {@link CLBuffer} object filled with the data from the provided float array
     */
    public static CLBuffer createAndFillCLBuffer(CLContext context, int size, CLMemory.Mem flag, float[] array)
    {
        CLBuffer clBuffer = context.createFloatBuffer(size, flag);
        fillBufferWithFloatArray(clBuffer, array);

        return clBuffer;
    }


    /**
     * Fills an OpenCL buffer with data from the provided float array.
     *
     * @param clBuffer the OpenCL buffer to be filled with data
     * @param array    the float array containing the data to be copied into the buffer
     * @throws IndexOutOfBoundsException if the specified index is out of range
     */
    public static void fillBufferWithFloatArray(CLBuffer<FloatBuffer> clBuffer, float[] array)
    {
        FloatBuffer buffer = clBuffer.getBuffer();
        for(int n=0; n<array.length; n++) {
            buffer.put(n, array[n]);
        }
    }


    /**
     * Retrieves a resource as a string from the classpath.
     *
     * @param c            the class from which the resource is loaded
     * @param resourceName the name of the resource to be retrieved
     * @return the contents of the resource as a string, or an empty string if the resource is not found or an error occurs
     */
    public static String getResourceAsString(Class c, String resourceName)
    {
        InputStream programStream = c.getResourceAsStream("/" + resourceName);
        String programString = "";

        try {
            programString = inputStreamToString(programStream);
        } catch (IOException var5) {
            var5.printStackTrace();
        }

        return programString;
    }


    /**
     * Converts the contents of an InputStream to a String using UTF-8 encoding.
     *
     * @param inputStream the InputStream to be converted
     * @return a String containing the data read from the InputStream
     * @throws IOException if an I/O error occurs while reading from the InputStream
     */
    public static String inputStreamToString(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int length;
        while((length = inputStream.read(buffer)) != -1) {
            result.write(buffer, 0, length);
        }
        return result.toString("UTF-8");
    }


    /**
     * Replaces the first occurrence of a specified target substring in the source string with a replacement string.
     *
     * @param source      the original string in which the replacement will occur
     * @param target      the substring to be replaced
     * @param replacement the string to replace the target substring
     * @return a new string with the first occurrence of the target substring replaced by the replacement string;
     *         if the target substring is not found, the original string is returned unchanged
     */
    public static String replaceFirst(String source, String target, String replacement)
    {
        int index = source.indexOf(target);
        if (index == -1) {
            return source;
        }

        return source.substring(0, index)
                .concat(replacement)
                .concat(source.substring(index+target.length()));
    }


    /**
     * Rounds up the global size to the nearest multiple of the specified group size.
     *
     * @param groupSize  the size of each group
     * @param globalSize the total size to be rounded up
     * @return the rounded-up value of globalSize to the nearest multiple of groupSize
     */
    private static int roundUp(int groupSize, int globalSize)
    {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 2D ---- //
    // ----------------------------------------- //

    /**
     * Calculates the local statistics (mean and standard deviation) for a 2D image using OpenCL.
     *
     * @param context         the OpenCL context in which resources are created
     * @param device          the OpenCL device used for computations
     * @param queue           the command queue for executing OpenCL kernels
     * @param inputImage2D    the input 2D image for which local statistics are computed
     * @param blockWidth      the width of the block used for the analysis
     * @param blockHeight     the height of the block used for the analysis
     * @param EPSILON         a small value to prevent division by zero in calculations
     * @return a {@link CLLocalStatistics} object containing the calculated local means and standard deviations,
     *         as well as the OpenCL buffers used during computation
     */
    public static CLLocalStatistics getLocalStatistics2D(CLContext context, CLDevice device, CLCommandQueue queue,
                                                         Utils.InputImage2D inputImage2D, int blockWidth,
                                                         int blockHeight, float EPSILON)
    {

        IJ.log("Calculating local statistics...");

        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int bRW = blockWidth/2;
        int bRH = blockHeight/2;

        // Get final block size (after removing pixels outside inbound circle/ellipse)
        int blockSize = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float)(x-bRW);
                float dy = (float)(y-bRH);
                if (((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clImageArray = CLUtils.createAndFillCLBuffer(context, imageSize, READ_ONLY, inputImage2D.getImageArray());
        queue.putWriteBuffer(clImageArray, true);

        float[] localMeans = new float[inputImage2D.getSize()];
        CLBuffer<FloatBuffer> clLocalMeans = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE, localMeans);
        queue.putWriteBuffer(clLocalMeans, true);

        float[] localStds = new float[inputImage2D.getSize()];
        CLBuffer<FloatBuffer> clLocalStds = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE, localStds);
        queue.putWriteBuffer(clLocalStds, true);

        // Create OpenCL program
        String programStringGetLocalStatistics2D = getResourceAsString(CLUtils.class, "kernelGetLocalStatistics2D.cl");
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$WIDTH$", "" + imageWidth);
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$HEIGHT$", "" + imageHeight);
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$BRW$", "" + bRW);
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$BRH$", "" + bRH);
        programStringGetLocalStatistics2D = replaceFirst(programStringGetLocalStatistics2D, "$EPSILON$", "" + EPSILON);
        CLProgram programGetLocalStatistics2D = context.createProgram(programStringGetLocalStatistics2D).build();

        // Crete OpenCL kernel and set args
        CLKernel kernelGetLocalStatistics2D = programGetLocalStatistics2D.createCLKernel("kernelGetLocalStatistics2D");

        int argn = 0;
        kernelGetLocalStatistics2D.setArg(argn++, clImageArray);
        kernelGetLocalStatistics2D.setArg(argn++, clLocalMeans);
        kernelGetLocalStatistics2D.setArg(argn++, clLocalStds);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetLocalStatistics2D,
                                0,
                                0,
                                imageWidth,
                                imageHeight,
                                0,
                                0);
        queue.finish();

        // Read the local means map back from the device
        queue.putReadBuffer(clLocalMeans, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                localMeans[y*imageWidth+x] = clLocalMeans.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Read the local stds map back from the device
        queue.putReadBuffer(clLocalStds, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                localStds[y*imageWidth+x] = clLocalStds.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Release memory
        kernelGetLocalStatistics2D.release();
        programGetLocalStatistics2D.release();

        return new CLLocalStatistics(localMeans, localStds, clImageArray, clLocalMeans, clLocalStds);
    }


    /**
     * Computes the block-wise Pearson correlation for a 2D image using OpenCL.
     *
     * This method calculates the Pearson correlation coefficient between a reference block
     * and all blocks of the input image. The result is stored in a repetition map, indicating
     * the degree of correlation across the image. Optionally, a relevance mask can be applied
     * to filter the results based on local standard deviations and a relevance constant.
     *
     * @param inputImage2D      the input 2D image for correlation calculation
     * @param referenceBlock2D  the reference block used for comparison in the correlation calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the Pearson correlation
     *         coefficients across the image
     */
    public static float[] getBlockPearson2D(Utils.InputImage2D inputImage2D,
                                            Utils.ReferenceBlock2D referenceBlock2D,
                                            float relevanceConstant,
                                            boolean normalizeOutput,
                                            boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int blockWidth = referenceBlock2D.getWidth();
        int blockHeight = referenceBlock2D.getHeight();
        int blockSize = referenceBlock2D.getSize();
        int bRW = referenceBlock2D.getRadiusWidth();
        int bRH = referenceBlock2D.getRadiusHeight();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(context, device, queue, inputImage2D,
                blockWidth, blockHeight, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock2D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockPearson2D = getResourceAsString(CLUtils.class, "kernelGetBlockPearson2D.cl");
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BLOCK_WIDTH$", "" + blockWidth);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BLOCK_HEIGHT$", "" + blockHeight);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BRW$", "" + bRW);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BRH$", "" + bRH);
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BLOCK_MEAN$", "" + referenceBlock2D.getMean());
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$BLOCK_STD$", "" + referenceBlock2D.getStd());
        programStringGetBlockPearson2D = replaceFirst(programStringGetBlockPearson2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockPearson2D = context.createProgram(programStringGetBlockPearson2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockPearson2D = programGetBlockPearson2D.createCLKernel("kernelGetBlockPearson2D");

        int argn = 0;
        kernelGetBlockPearson2D.setArg(argn++, clBlockPixels);
        kernelGetBlockPearson2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockPearson2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockPearson2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockPearson2D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetBlockPearson2D, 0, 0, imageWidth, imageHeight,
                0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                repetitionMap[y*imageWidth+x] = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Calculate and apply relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage2D.getImageArray(),
                imageWidth, imageHeight, bRW, bRH, localStatistics.getLocalStds(), relevanceConstant);
        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH, relevanceMask.getRelevanceMask());
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise cosine similarity for a 2D image using OpenCL.
     *
     * This method calculates the cosine similarity between a reference block and all blocks
     * of the input image. The result is stored in a repetition map, indicating the degree
     * of similarity across the image. Optionally, a relevance mask can be applied to filter
     * the results based on local standard deviations and a relevance constant.
     *
     * @param inputImage2D      the input 2D image for similarity calculation
     * @param referenceBlock2D  the reference block used for comparison in the similarity calculation
     * @param relevanceConstant  a constant value used to cintrol the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the cosine similarity
     *         coefficients across the image
     */
    public static float[] getBlockCosineSimilarity2D(Utils.InputImage2D inputImage2D,
                                                     Utils.ReferenceBlock2D referenceBlock2D,
                                                     float relevanceConstant,
                                                     boolean normalizeOutput,
                                                     boolean useDevice)
    {
        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int blockWidth = referenceBlock2D.getWidth();
        int blockHeight = referenceBlock2D.getHeight();
        int blockSize = referenceBlock2D.getSize();
        int bRW = referenceBlock2D.getRadiusWidth();
        int bRH = referenceBlock2D.getRadiusHeight();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = getLocalStatistics2D(context, device, queue, inputImage2D,
                blockWidth, blockHeight, Utils.EPSILON);

        // Create and fill OpenCL buffers
        IJ.log("Calculating Structural Repetition Scores...");

        CLBuffer<FloatBuffer> clBlockPixels = createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock2D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockCosineSimilarity2D = getResourceAsString(CLUtils.class, "kernelGetBlockCosineSimilarity2D.cl");
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$BRW$", "" + bRW);
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$BRH$", "" + bRH);
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$BLOCK_STD$", "" + referenceBlock2D.getStd());
        programStringGetBlockCosineSimilarity2D = replaceFirst(programStringGetBlockCosineSimilarity2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockCosineSimilarity2D = context.createProgram(programStringGetBlockCosineSimilarity2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockCosineSimilarity2D = programGetBlockCosineSimilarity2D.createCLKernel("kernelGetBlockCosineSimilarity2D");

        int argn = 0;
        kernelGetBlockCosineSimilarity2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockCosineSimilarity2D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetBlockCosineSimilarity2D, 0, 0, imageWidth,
                imageHeight, 0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                repetitionMap[y*imageWidth+x] = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Calculate and apply relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage2D.getImageArray(), imageWidth,
                imageHeight, bRW, bRH, localStatistics.getLocalStds(), relevanceConstant);
        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH,
                    relevanceMask.getRelevanceMask());
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise Structural Similarity Index (SSIM) for a 2D image using OpenCL.
     *
     * This method calculates the SSIM between a reference block and all blocks of the input image,
     * generating a repetition map that indicates the similarity strength across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local
     * standard deviations and a relevance constant.
     *
     * @param inputImage2D      the input 2D image for SSIM calculation
     * @param referenceBlock2D  the reference block used for comparison in the SSIM calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the SSIM coefficients
     *         across the image
     */
    public static float[] getBlockSsim2D(Utils.InputImage2D inputImage2D,
                                            Utils.ReferenceBlock2D referenceBlock2D,
                                            float relevanceConstant,
                                            boolean normalizeOutput,
                                            boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int blockWidth = referenceBlock2D.getWidth();
        int blockHeight = referenceBlock2D.getHeight();
        int blockSize = referenceBlock2D.getSize();
        int bRW = referenceBlock2D.getRadiusWidth();
        int bRH = referenceBlock2D.getRadiusHeight();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(context, device, queue, inputImage2D,
                blockWidth, blockHeight, Utils.EPSILON);


        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock2D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockSsim2D = getResourceAsString(CLUtils.class, "kernelGetBlockSsim2D.cl");
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BW$", "" + blockWidth);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BH$", "" + blockHeight);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BRW$", "" + bRW);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BRH$", "" + bRH);
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BLOCK_MEAN$", "" + referenceBlock2D.getMean());
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$BLOCK_STD$", "" + referenceBlock2D.getStd());
        programStringGetBlockSsim2D = replaceFirst(programStringGetBlockSsim2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockSsim2D = context.createProgram(programStringGetBlockSsim2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockSsim2D = programGetBlockSsim2D.createCLKernel("kernelGetBlockSsim2D");

        int argn = 0;
        kernelGetBlockSsim2D.setArg(argn++, clBlockPixels);
        kernelGetBlockSsim2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockSsim2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockSsim2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockSsim2D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetBlockSsim2D, 0, 0, imageWidth, imageHeight,
                0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                repetitionMap[y*imageWidth+x] = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Calculate and apply relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage2D.getImageArray(), imageWidth,
                imageHeight, bRW, bRH, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH,
                    relevanceMask.getRelevanceMask());
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise Normalized Root Mean Square Error (NRMSE) for a 2D image using OpenCL.
     *
     * This method calculates the NRMSE between a reference block and blocks of the input image,
     * generating a repetition map that indicates the similarity (inverted NRMSE) across the image.
     * It allows for optional relevance masking based on local standard deviations and a relevance constant.
     *
     * @param inputImage2D      the input 2D image for NRMSE calculation
     * @param referenceBlock2D  the reference block used for comparison in the NRMSE calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the inverted NRMSE coefficients
     *         across the image (higher values indicate greater similarity)
     */
    public static float[] getBlockNrmse2D(Utils.InputImage2D inputImage2D,
                                          Utils.ReferenceBlock2D referenceBlock2D,
                                          float relevanceConstant,
                                          boolean normalizeOutput,
                                          boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int blockWidth = referenceBlock2D.getWidth();
        int blockHeight = referenceBlock2D.getHeight();
        int blockSize = referenceBlock2D.getSize();
        int bRW = referenceBlock2D.getRadiusWidth();
        int bRH = referenceBlock2D.getRadiusHeight();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(context, device, queue, inputImage2D,
                blockWidth, blockHeight, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock2D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockNrmse2D = getResourceAsString(CLUtils.class, "kernelGetBlockNrmse2D.cl");
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BW$", "" + blockWidth);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BH$", "" + blockHeight);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BRW$", "" + bRW);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BRH$", "" + bRH);
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$BLOCK_MEAN$", "" + referenceBlock2D.getMean());
        programStringGetBlockNrmse2D = replaceFirst(programStringGetBlockNrmse2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockNrmse2D = context.createProgram(programStringGetBlockNrmse2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockNrmse2D = programGetBlockNrmse2D.createCLKernel("kernelGetBlockNrmse2D");

        int argn = 0;
        kernelGetBlockNrmse2D.setArg(argn++, clBlockPixels);
        kernelGetBlockNrmse2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockNrmse2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockNrmse2D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetBlockNrmse2D, 0, 0,
                imageWidth, imageHeight,
                0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int y=bRH; y<imageHeight-bRH; y++) {
            for (int x=bRW; x<imageWidth-bRW; x++) {
                repetitionMap[y*imageWidth+x] = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Calculate and apply relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage2D.getImageArray(), imageWidth,
                imageHeight, bRW, bRH, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH,
                relevanceMask.getRelevanceMask());

        // Invert NRMSE
        for (int y=bRH; y<imageHeight-bRH; y++) {
            for (int x=bRW; x<imageWidth-bRW; x++) {
                int index = y*imageWidth+x;
                if(relevanceMask.getRelevanceMask()[index]>0.0f) {
                    float rmse = repetitionMap[index];
                    if (rmse == 0.0f) { // Special case where RMSE is 0, 1/rmse would be undefined but we want perfect similarity
                        repetitionMap[index] = 1.0f;
                    } else {
                        repetitionMap[index] = 1.0f / rmse;
                    }
                }
            }
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH,
                    relevanceMask.getRelevanceMask());
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise absolute difference of standard deviations for a 2D image using OpenCL.
     *
     * This method calculates the absolute difference between the standard deviation of a reference block
     * and the local standard deviations of the input image. The resulting repetition map indicates the
     * dissimilarity (inverted similarity) across the image, allowing for optional relevance masking
     * based on local standard deviations and a relevance constant.
     *
     * @param inputImage2D      the input 2D image for dissimilarity calculation
     * @param referenceBlock2D  the reference block used for comparison in the dissimilarity calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the similarity based on
     *         absolute differences in standard deviations (higher values indicate greater similarity)
     */
    public static float[] getBlockAbsDiffStds2D(Utils.InputImage2D inputImage2D,
                                                     Utils.ReferenceBlock2D referenceBlock2D,
                                                     float relevanceConstant,
                                                     boolean normalizeOutput,
                                                     boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage2D.getWidth();
        int imageHeight = inputImage2D.getHeight();
        int imageSize = inputImage2D.getSize();
        int blockWidth = referenceBlock2D.getWidth();
        int blockHeight = referenceBlock2D.getHeight();
        int blockSize = referenceBlock2D.getSize();
        int bRW = referenceBlock2D.getRadiusWidth();
        int bRH = referenceBlock2D.getRadiusHeight();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(context, device, queue, inputImage2D,
                blockWidth, blockHeight, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock2D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockAbsDiffStds2D = getResourceAsString(CLUtils.class, "kernelGetBlockAbsDiffStds2D.cl");
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$BRW$", "" + bRW);
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$BRH$", "" + bRH);
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$BLOCK_STD$", "" + referenceBlock2D.getStd());
        programStringGetBlockAbsDiffStds2D = replaceFirst(programStringGetBlockAbsDiffStds2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockAbsDiffStds2D = context.createProgram(programStringGetBlockAbsDiffStds2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockAbsDiffStds2D = programGetBlockAbsDiffStds2D.createCLKernel("kernelGetBlockAbsDiffStds2D");

        int argn = 0;
        kernelGetBlockAbsDiffStds2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockAbsDiffStds2D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put2DRangeKernel(kernelGetBlockAbsDiffStds2D, 0, 0, imageWidth,
                imageHeight, 0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int y=0; y<imageHeight; y++) {
            for (int x=0; x<imageWidth; x++) {
                repetitionMap[y*imageWidth+x] = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                queue.finish();
            }
        }

        // Normalize to range
        float repetitionMin = Float.MAX_VALUE;
        float repetitionMax = Float.MIN_VALUE;

        for (int y=bRH; y<imageHeight-bRH; y++) {
            for (int x=bRW; x<imageWidth-bRW; x++) {
                repetitionMin = min(repetitionMin, repetitionMap[y*imageWidth+x]);
                repetitionMax = max(repetitionMax, repetitionMap[y*imageWidth+x]);
            }
        }

        for (int y=bRH; y<imageHeight-bRH; y++) {
            for (int x=bRW; x<imageWidth-bRW; x++) {
                repetitionMap[y*imageWidth+x] = (repetitionMap[y*imageWidth+x]-repetitionMin)/(repetitionMax-repetitionMin+Utils.EPSILON);
            }
        }

        // Convert dissimilarity into similarity
        for (int y=bRH; y<imageHeight-bRH; y++) {
            for (int x=bRW; x<imageWidth-bRW; x++) {
                repetitionMap[y*imageWidth+x] = 1.0f - repetitionMap[y*imageWidth+x];
            }
        }

        // Calculate and apply relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage2D.getImageArray(), imageWidth,
                imageHeight, bRW, bRH, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, bRW, bRH,
                    relevanceMask.getRelevanceMask());
        }


        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Calculate the 2D repetition map based on the selected metric.
     *
     * @param metric The selected metric for repetition calculation.
     * @param inputImage The input image data.
     * @param referenceBlock The reference block for comparison.
     * @param relevanceConstant The relevance constant for calculation.
     * @param normalizeOutput Whether to normalize the output.
     * @param useDevice Whether to use the device from preferences.
     * @return The calculated repetition map.
     */
    public static float[] calculateBlockRepetitionMap2D(String metric, Utils.InputImage2D inputImage,
                                                        Utils.ReferenceBlock2D referenceBlock, float relevanceConstant,
                                                        boolean normalizeOutput, boolean useDevice)
    {
        if (metric.equals(BlockRepetition2D_.METRICS[0])) {
            return CLUtils.getBlockPearson2D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition2D_.METRICS[1])) {
            return CLUtils.getBlockCosineSimilarity2D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition2D_.METRICS[2])) {
            return CLUtils.getBlockSsim2D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition2D_.METRICS[3])) {
            return CLUtils.getBlockNrmse2D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition2D_.METRICS[4])) {
            return CLUtils.getBlockAbsDiffStds2D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else {
            return null; // This is here just because the method requires a return statement outside the IF/ELSE clauses.
        }
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 3D ---- //
    // ----------------------------------------- //

    /**
     * Calculates the local statistics (mean and standard deviation) for a 3D image using OpenCL.
     *
     * @param openCLResources an {@link OpenCLResources} object
     * @param inputImage3D    the input 3D image for which local statistics are computed, given as a flattened array
     * @param blockSize       the block size after removing pixels outside the inbound sphere/spheroid
     * @param blockRadiusWidth   the block radius along the width
     * @param blockRadiusHeight  the block radius along the height
     * @param blockRadiusDepth   the block radius along the depth
     * @param EPSILON         a small value to prevent division by zero in calculations
     * @return a {@link CLLocalStatistics} object containing the calculated local means and standard deviations,
     *         as well as the OpenCL buffers used during computation
     */
    public static CLLocalStatistics getLocalStatistics3D(OpenCLResources openCLResources,
                                                         Utils.InputImage3D inputImage3D,
                                                         int blockSize, int blockRadiusWidth, int blockRadiusHeight,
                                                         int blockRadiusDepth, float EPSILON)
    {

        IJ.log("Calculating local statistics...");

        // Cache variables
        int imageWidth = inputImage3D.getWidth();
        int imageHeight = inputImage3D.getHeight();
        int imageDepth = inputImage3D.getDepth();
        int imageSize = inputImage3D.getSize();

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clImageArray = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize, READ_ONLY, inputImage3D.getImageArray());
        openCLResources.getQueue().putWriteBuffer(clImageArray, true);

        float[] localMeans = new float[inputImage3D.getSize()];
        CLBuffer<FloatBuffer> clLocalMeans = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize, READ_WRITE, localMeans);
        openCLResources.getQueue().putWriteBuffer(clLocalMeans, true);

        float[] localStds = new float[inputImage3D.getSize()];
        CLBuffer<FloatBuffer> clLocalStds = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize, READ_WRITE, localStds);
        openCLResources.getQueue().putWriteBuffer(clLocalStds, true);

        // Create OpenCL program
        String programStringGetLocalStatistics3D = getResourceAsString(CLUtils.class, "kernelGetLocalStatistics3D.cl");
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$WIDTH$", "" + imageWidth);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$HEIGHT$", "" + imageHeight);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$DEPTH$", "" + imageDepth);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetLocalStatistics3D = replaceFirst(programStringGetLocalStatistics3D, "$EPSILON$", "" + EPSILON);
        CLProgram programGetLocalStatistics3D = openCLResources.getContext().createProgram(programStringGetLocalStatistics3D).build();

        // Crete OpenCL kernel and set args
        CLKernel kernelGetLocalStatistics3D = programGetLocalStatistics3D.createCLKernel("kernelGetLocalStatistics3D");

        int argn = 0;
        kernelGetLocalStatistics3D.setArg(argn++, clImageArray);
        kernelGetLocalStatistics3D.setArg(argn++, clLocalMeans);
        kernelGetLocalStatistics3D.setArg(argn++, clLocalStds);

        int localWorkSize = min(openCLResources.getDevice().getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        openCLResources.getQueue().put3DRangeKernel(kernelGetLocalStatistics3D,
                0, 0, 0,
                imageWidth, imageHeight, imageDepth,
                0, 0, 0);
        openCLResources.getQueue().finish();

        // Read the local means map back from the device
        openCLResources.getQueue().putReadBuffer(clLocalMeans, true);
        for(int z=0; z<imageDepth; z++){
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    localMeans[index] = clLocalMeans.getBuffer().get(index);
                }
            }
        }
        openCLResources.getQueue().finish();

        // Read the local stds map back from the device
        openCLResources.getQueue().putReadBuffer(clLocalStds, true);
        for(int z=0; z<imageDepth; z++){
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    localStds[index] = clLocalStds.getBuffer().get(index);
                }
            }
        }
        openCLResources.getQueue().finish();

        // Release memory
        kernelGetLocalStatistics3D.release();
        programGetLocalStatistics3D.release();

        return new CLLocalStatistics(localMeans, localStds, clImageArray, clLocalMeans, clLocalStds);
    }


    /**
     * Computes the block-wise Pearson correlation for a 3D image using OpenCL.
     *
     * This method calculates the Pearson correlation coefficient between a reference block
     * and all blocks of the input image. The result is stored in a repetition map, indicating
     * the degree of correlation across the image. Optionally, a relevance mask can be applied
     * to filter the results based on local standard deviations and a relevance constant.
     *
     * @param inputImage3D      the input 3D image for correlation calculation, provided as a flattened 1-D array
     * @param referenceBlock3D  the reference block used for comparison in the correlation calculation, provided as a flattened 1-D array
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the Pearson correlation
     *         coefficients across the image
     */
    public static float[] getBlockPearson3D(Utils.InputImage3D inputImage3D, Utils.ReferenceBlock3D referenceBlock3D,
                                            float relevanceConstant, boolean normalizeOutput, boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");
        // TODO: NEED TO SOLVE ISSUE HERE WHERE REPETITION MAP HAS STRIDING ARTEFACTS WITH BLANK PIXELS (OVERFLOW?)
        // Cache variables
        int imageWidth = inputImage3D.getWidth();
        int imageHeight = inputImage3D.getHeight();
        int imageDepth = inputImage3D.getDepth();
        int imageSize = inputImage3D.getSize();
        int blockWidth = referenceBlock3D.getWidth();
        int blockHeight = referenceBlock3D.getHeight();
        int blockDepth = referenceBlock3D.getDepth();
        int blockSize = referenceBlock3D.getSize();
        int blockRadiusWidth = referenceBlock3D.getRadiusWidth();
        int blockRadiusHeight = referenceBlock3D.getRadiusHeight();
        int blockRadiusDepth = referenceBlock3D.getDepth();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics3D(openCLResources, inputImage3D, blockSize,
                blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock3D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockPearson3D = getResourceAsString(CLUtils.class, "kernelGetBlockPearson3D.cl");
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$DEPTH$", "" + imageDepth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_WIDTH$", "" + blockWidth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_HEIGHT$", "" + blockHeight);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_DEPTH$", "" + blockDepth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_MEAN$", "" + referenceBlock3D.getMean());
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$BLOCK_STD$", "" + referenceBlock3D.getStd());
        programStringGetBlockPearson3D = replaceFirst(programStringGetBlockPearson3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockPearson3D = context.createProgram(programStringGetBlockPearson3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockPearson3D = programGetBlockPearson3D.createCLKernel("kernelGetBlockPearson3D");

        int argn = 0;
        kernelGetBlockPearson3D.setArg(argn++, clBlockPixels);
        kernelGetBlockPearson3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockPearson3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockPearson3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockPearson3D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put3DRangeKernel(kernelGetBlockPearson3D,
                0, 0, 0,
                imageWidth, imageHeight, imageDepth,
                0, 0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for (int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    repetitionMap[index] = clRepetitionMap.getBuffer().get(index);
                    queue.finish();
                }
            }
        }

        // Calculate and apply relevance mask
        float[] relevanceMask = Utils.getRelevanceMask3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                blockRadiusHeight, blockRadiusDepth, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth, relevanceMask);
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask);
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise cosine similarity for a 3D image using OpenCL.
     *
     * This method calculates the cosine similarity between a reference block and all blocks
     * of the input image. The result is stored in a repetition map, indicating the degree
     * of similarity across the image. Optionally, a relevance mask can be applied to filter
     * the results based on local standard deviations and a relevance constant.
     *
     * @param inputImage3D      the input 3D image for similarity calculation
     * @param referenceBlock3D  the reference block used for comparison in the similarity calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the cosine similarity
     *         coefficients across the image
     */
    public static float[] getBlockCosineSimilarity3D(Utils.InputImage3D inputImage3D,
                                                     Utils.ReferenceBlock3D referenceBlock3D, float relevanceConstant,
                                                     boolean normalizeOutput, boolean useDevice)
    {
        // Cache variables
        int imageWidth = inputImage3D.getWidth();
        int imageHeight = inputImage3D.getHeight();
        int imageSize = inputImage3D.getSize();
        int imageDepth = inputImage3D.getDepth();
        int blockSize = referenceBlock3D.getSize();
        int blockRadiusWidth = referenceBlock3D.getRadiusWidth();
        int blockRadiusHeight = referenceBlock3D.getRadiusHeight();
        int blockRadiusDepth = referenceBlock3D.getRadiusDepth();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = getLocalStatistics3D(openCLResources, inputImage3D, blockSize,
                blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, Utils.EPSILON);

        // Create and fill OpenCL buffers
        IJ.log("Calculating Structural Repetition Scores...");

        CLBuffer<FloatBuffer> clBlockPixels = createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock3D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize]; // Create array to hold the repetition map
        Arrays.fill(repetitionMap, 0.0f); // Fill with zeroes just be sure

        CLBuffer<FloatBuffer> clRepetitionMap = createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockCosineSimilarity3D = getResourceAsString(CLUtils.class, "kernelGetBlockCosineSimilarity3D.cl");
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$DEPTH$", "" + imageDepth);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$BLOCK_STD$", "" + referenceBlock3D.getStd());
        programStringGetBlockCosineSimilarity3D = replaceFirst(programStringGetBlockCosineSimilarity3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockCosineSimilarity3D = context.createProgram(programStringGetBlockCosineSimilarity3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockCosineSimilarity3D = programGetBlockCosineSimilarity3D.createCLKernel("kernelGetBlockCosineSimilarity3D");

        int argn = 0;
        kernelGetBlockCosineSimilarity3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockCosineSimilarity3D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);
        int globalWorkSize = roundUp(localWorkSize, imageSize);

        queue.put3DRangeKernel(kernelGetBlockCosineSimilarity3D,
                0, 0, 0,
                imageWidth, imageHeight, imageDepth,
                0, 0, 0);

        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    repetitionMap[index] = clRepetitionMap.getBuffer().get(index);
                    queue.finish();
                }
            }
        }

        // Calculate and apply relevance mask
        float[] relevanceMask = Utils.getRelevanceMask3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                blockRadiusHeight, blockRadiusDepth, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth, relevanceMask);
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask);
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise Structural Similarity Index (SSIM) for a 3D image using OpenCL.
     *
     * This method calculates the SSIM between a reference block and all blocks of the input image,
     * generating a repetition map that indicates the similarity strength across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local
     * standard deviations and a relevance constant.
     *
     * @param inputImage3D      the input 3D image for SSIM calculation
     * @param referenceBlock3D  the reference block used for comparison in the SSIM calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the SSIM coefficients
     *         across the image
     */
    public static float[] getBlockSsim3D(Utils.InputImage3D inputImage3D, Utils.ReferenceBlock3D referenceBlock3D,
                                         float relevanceConstant, boolean normalizeOutput, boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage3D.getWidth();
        int imageHeight = inputImage3D.getHeight();
        int imageDepth = inputImage3D.getDepth();
        int imageSize = inputImage3D.getSize();
        int blockWidth = referenceBlock3D.getWidth();
        int blockHeight = referenceBlock3D.getHeight();
        int blockDepth = referenceBlock3D.getDepth();
        int blockSize = referenceBlock3D.getSize();
        int blockRadiusWidth = referenceBlock3D.getRadiusWidth();
        int blockRadiusHeight = referenceBlock3D.getRadiusHeight();
        int blockRadiusDepth = referenceBlock3D.getRadiusDepth();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics3D(openCLResources, inputImage3D, blockSize,
                blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock3D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        Arrays.fill(repetitionMap, 0.0f);
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockSsim3D = getResourceAsString(CLUtils.class, "kernelGetBlockSsim3D.cl");
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$DEPTH$", "" + imageDepth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BW$", "" + blockWidth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BH$", "" + blockHeight);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BZ$", "" + blockDepth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BRW$", "" + blockWidth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BLOCK_MEAN$", "" + referenceBlock3D.getMean());
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$BLOCK_STD$", "" + referenceBlock3D.getStd());
        programStringGetBlockSsim3D = replaceFirst(programStringGetBlockSsim3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockSsim3D = context.createProgram(programStringGetBlockSsim3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockSsim3D = programGetBlockSsim3D.createCLKernel("kernelGetBlockSsim3D");

        int argn = 0;
        kernelGetBlockSsim3D.setArg(argn++, clBlockPixels);
        kernelGetBlockSsim3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockSsim3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockSsim3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetBlockSsim3D.setArg(argn++, clRepetitionMap);

        int localWorkSize = min(device.getMaxWorkGroupSize(), 256);

        queue.put3DRangeKernel(kernelGetBlockSsim3D, 0, 0, 0,
                imageWidth, imageHeight, imageDepth,
                0, 0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    repetitionMap[index] = clRepetitionMap.getBuffer().get(index);
                    queue.finish();
                }
            }
        }

        // Calculate and apply relevance mask
        float[] relevanceMask = Utils.getRelevanceMask3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                blockRadiusHeight, blockRadiusDepth, localStatistics.getLocalStds(), relevanceConstant);

        if(relevanceConstant>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth, relevanceMask);
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, relevanceMask);
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Computes the block-wise Normalized Root Mean Square Error (NRMSE) for a 3D image using OpenCL.
     *
     * This method calculates the NRMSE between a reference block and blocks of the input image,
     * generating a repetition map that indicates the similarity (inverted NRMSE) across the image.
     * It allows for optional relevance masking based on local standard deviations and a relevance constant.
     *
     * @param inputImage3D      the input 3D image for NRMSE calculation
     * @param referenceBlock3D  the reference block used for comparison in the NRMSE calculation
     * @param relevanceConstant  a constant value used to control the strength of the structural relevance filter
     * @param normalizeOutput    a boolean indicating whether to normalize the output repetition map
     * @param useDevice          a boolean indicating whether to use the user-defined OpenCL device preference
     * @return a float array representing the repetition map, indicating the inverted NRMSE coefficients
     *         across the image (higher values indicate greater similarity)
     */
    public static float[] getBlockNrmse3D(Utils.InputImage3D inputImage3D, Utils.ReferenceBlock3D referenceBlock3D,
                                          float relevanceConstant, boolean normalizeOutput, boolean useDevice)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage3D.getWidth();
        int imageHeight = inputImage3D.getHeight();
        int imageDepth = inputImage3D.getDepth();
        int imageSize = inputImage3D.getSize();
        int blockSize = referenceBlock3D.getSize();
        int blockRadiusWidth = referenceBlock3D.getRadiusWidth();
        int blockRadiusHeight = referenceBlock3D.getRadiusHeight();
        int blockRadiusDepth = referenceBlock3D.getRadiusDepth();

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);

        // Retrieve OpenCL context, device and queue
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLLocalStatistics localStatistics = CLUtils.getLocalStatistics3D(openCLResources, inputImage3D, blockSize,
                blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, Utils.EPSILON);

        // Create and fill OpenCL buffers
        CLBuffer<FloatBuffer> clBlockPixels = CLUtils.createAndFillCLBuffer(context, blockSize, READ_ONLY,
                referenceBlock3D.getPixels());
        queue.putWriteBuffer(clBlockPixels, true);

        float[] repetitionMap = new float[imageSize];
        Arrays.fill(repetitionMap, 0.0f);
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(context, imageSize, READ_WRITE,
                repetitionMap);

        // Build OpenCL program
        String programStringGetBlockNrmse3D = getResourceAsString(CLUtils.class, "kernelGetBlockNrmse3D.cl");
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$WIDTH$", "" + imageWidth);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$HEIGHT$", "" + imageHeight);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$DEPTH$", "" + imageDepth);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetBlockNrmse3D = replaceFirst(programStringGetBlockNrmse3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetBlockNrmse3D = context.createProgram(programStringGetBlockNrmse3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetBlockNrmse3D = programGetBlockNrmse3D.createCLKernel("kernelGetBlockNrmse3D");

        int argn = 0;
        kernelGetBlockNrmse3D.setArg(argn++, clBlockPixels);
        kernelGetBlockNrmse3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetBlockNrmse3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetBlockNrmse3D.setArg(argn++, clRepetitionMap);

        queue.put3DRangeKernel(kernelGetBlockNrmse3D, 0, 0, 0,
                imageWidth, imageHeight, imageDepth,
                0, 0, 0);
        queue.finish();

        // Read the repetition map back from the device
        queue.putReadBuffer(clRepetitionMap, true);
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    repetitionMap[index] = clRepetitionMap.getBuffer().get(index);
                    queue.finish();
                }
            }
        }

        // Invert NRMSE // TODO: NOT SURE WHY INVERTING IS GIVING THE NON INVERTED RESULT -.-'
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float rmse = repetitionMap[index];
                    if (rmse == 0.0f) { // Special case where RMSE is 0 (i.e., 1/rmse is undefined), but we want perfect similarity
                        repetitionMap[index] = 1.0f;
                    } else {
                        repetitionMap[index] = 1.0f / rmse;
                    }
                }
            }
        }

        // Calculate and apply relevance mask
        float[] relevanceMask = new float[imageSize];
        if(relevanceConstant>0.0f) {
            relevanceMask = Utils.getRelevanceMask3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, localStatistics.getLocalStds(), relevanceConstant);

            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth, relevanceMask);
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            if(relevanceConstant>0.0f) {
                repetitionMap = Utils.normalizeImage3D(repetitionMap,
                        imageWidth, imageHeight, imageDepth, blockRadiusWidth, blockRadiusHeight, blockRadiusDepth,
                        relevanceMask);
            }else{
                repetitionMap = Utils.normalizeImage3D(repetitionMap,
                        imageWidth, imageHeight, imageDepth, blockRadiusWidth, blockRadiusHeight, blockRadiusDepth,
                        null);
            }
        }

        // Release memory
        context.release();

        return repetitionMap;
    }


    /**
     * Calculate the 3D repetition map based on the selected metric.
     *
     * @param metric The selected metric for repetition calculation.
     * @param inputImage The input image data.
     * @param referenceBlock The reference block for comparison.
     * @param relevanceConstant The relevance constant for calculation.
     * @param normalizeOutput Whether to normalize the output.
     * @param useDevice Whether to use the device from preferences.
     * @return The calculated repetition map.
     */
    public static float[] calculateBlockRepetitionMap3D(String metric, Utils.InputImage3D inputImage,
                                                        Utils.ReferenceBlock3D referenceBlock, float relevanceConstant,
                                                        boolean normalizeOutput, boolean useDevice)
    {
        if (metric.equals(BlockRepetition3D_.METRICS[0])) {
            return CLUtils.getBlockPearson3D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition3D_.METRICS[1])) {
            return CLUtils.getBlockCosineSimilarity3D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition3D_.METRICS[2])) {
            return CLUtils.getBlockSsim3D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else if (metric.equals(BlockRepetition3D_.METRICS[3])) {
            return CLUtils.getBlockNrmse3D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        //} else if (metric.equals(BlockRepetition3D_.METRICS[4])) {
            //return CLUtils.getBlockAbsDiffStds3D(inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);
        } else {
            return null; // This is here just because the method requires a return statement outside the IF clauses.
        }
    }


    // ------------------------------------------ //
    // ---- METHODS FOR GLOBAL REPETITION 2D ---- //
    // ------------------------------------------ //

    /**
     * Computes the Global Repetition Map based on Pearson correlations for a 2D image using OpenCL.
     *
     * This method calculates the global repetition map based on Pearson correlations. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage2D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockSize         the size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalPearson2D(Utils.InputImage2D inputImage, CLLocalStatistics localStatistics,
                                             int blockWidth, int blockHeight, int blockSize,
                                             Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                            OpenCLResources openCLResources
    ){
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalPearson2D = getResourceAsString(CLUtils.class, "kernelGetGlobalPearson2D.cl");
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalPearson2D = replaceFirst(programStringGetGlobalPearson2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalPearson2D = openCLResources.getContext().createProgram(programStringGetGlobalPearson2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalPearson2D = programGetGlobalPearson2D.createCLKernel("kernelGetGlobalPearson2D");

        int argn = 0;
        kernelGetGlobalPearson2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalPearson2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalPearson2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalPearson2D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalPearson2D.setArg(argn++, clRepetitionMap);


        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth/64 + ((imageWidth%64==0)?0:1);
        int nYBlocks = imageHeight/64 + ((imageHeight%64==0)?0:1);
        int totalBlocks = nXBlocks * nYBlocks; // Total number of blocks

        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, imageHeight-nYB*64);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, imageWidth-nXB*64);
                float progressPercentage = ((nYB*nXBlocks+nXB)/(float)totalBlocks)*100;
                showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                openCLResources.getQueue().put2DRangeKernel(kernelGetGlobalPearson2D, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                openCLResources.getQueue().finish();
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                float similarity = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                float weightSum = clWeightsSumMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                repetitionMap[y*imageWidth+x] = similarity/(weightSum*nPixels+Utils.EPSILON);
                openCLResources.getQueue().finish();
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, blockRadiusWidth,
                    blockRadiusHeight, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Cosine Similarity metric for a 2D image using OpenCL.
     *
     * This method calculates the global repetition map based on the Cosine Similarity metric. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage2D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalCosineSimilarity2D(Utils.InputImage2D inputImage, CLLocalStatistics localStatistics,
                                             int blockWidth, int blockHeight, Utils.RelevanceMask relevanceMask,
                                             float nPixels, boolean normalizeOutput, OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalCosineSimilarity2D = getResourceAsString(CLUtils.class, "kernelGetGlobalCosineSimilarity2D.cl");
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalCosineSimilarity2D = replaceFirst(programStringGetGlobalCosineSimilarity2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalCosineSimilarity2D = openCLResources.getContext().createProgram(programStringGetGlobalCosineSimilarity2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalCosineSimilarity2D = programGetGlobalCosineSimilarity2D.createCLKernel("kernelGetGlobalCosineSimilarity2D");

        int argn = 0;
        kernelGetGlobalCosineSimilarity2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalCosineSimilarity2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalCosineSimilarity2D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalCosineSimilarity2D.setArg(argn++, clRepetitionMap);

        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth/64 + ((imageWidth%64==0)?0:1);
        int nYBlocks = imageHeight/64 + ((imageHeight%64==0)?0:1);
        int totalBlocks = nXBlocks * nYBlocks; // Total number of blocks

        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, imageHeight-nYB*64);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, imageWidth-nXB*64);
                float progressPercentage = ((nYB*nXBlocks+nXB)/(float)totalBlocks)*100;
                showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                openCLResources.getQueue().put2DRangeKernel(kernelGetGlobalCosineSimilarity2D, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                openCLResources.getQueue().finish();
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                float similarity = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                float weightSum = clWeightsSumMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                repetitionMap[y*imageWidth+x] = similarity/(weightSum*nPixels+Utils.EPSILON);
                openCLResources.getQueue().finish();
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, blockRadiusWidth,
                    blockRadiusHeight, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Structural Similarity Index Metric (SSIM) for a 2D image using OpenCL.
     *
     * This method calculates the global repetition map based on the Structural Similarity Index Metric (SSIM). The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage2D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockSize         the size of the block (in pixels) after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an OpenCLResources object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalSsim2D(Utils.InputImage2D inputImage, CLLocalStatistics localStatistics,
                                          int blockWidth, int blockHeight, int blockSize,
                                          Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                          OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalSsim2D = getResourceAsString(CLUtils.class, "kernelGetGlobalSsim2D.cl");
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalSsim2D = replaceFirst(programStringGetGlobalSsim2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalSsim2D = openCLResources.getContext().createProgram(programStringGetGlobalSsim2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalSsim2D = programGetGlobalSsim2D.createCLKernel("kernelGetGlobalSsim2D");

        int argn = 0;
        kernelGetGlobalSsim2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalSsim2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalSsim2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalSsim2D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalSsim2D.setArg(argn++, clRepetitionMap);

        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth/64 + ((imageWidth%64==0)?0:1);
        int nYBlocks = imageHeight/64 + ((imageHeight%64==0)?0:1);
        int totalBlocks = nXBlocks * nYBlocks; // Total number of blocks

        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, imageHeight-nYB*64);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, imageWidth-nXB*64);
                float progressPercentage = ((nYB*nXBlocks+nXB)/(float)totalBlocks)*100;
                showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                openCLResources.getQueue().put2DRangeKernel(kernelGetGlobalSsim2D, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                openCLResources.getQueue().finish();
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                float similarity = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                float weightSum = clWeightsSumMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                repetitionMap[y*imageWidth+x] = similarity/(weightSum*nPixels+Utils.EPSILON);
                openCLResources.getQueue().finish();
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, blockRadiusWidth,
                    blockRadiusHeight, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Normalized Root Mean Squared Error (NRMSE) for a 2D image using OpenCL.
     *
     * This method calculates the global repetition map based on the (inverted) Normalized Root Mean Squared Error (NRMSE). The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage2D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockSize         the size of the block used for the anaysis, in pixels, after removing the pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally  relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalNrmse2D(Utils.InputImage2D inputImage, CLLocalStatistics localStatistics,
                                          int blockWidth, int blockHeight, int blockSize,
                                          Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                          OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalNrmse2D = getResourceAsString(CLUtils.class, "kernelGetGlobalNrmse2D.cl");
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalNrmse2D = replaceFirst(programStringGetGlobalNrmse2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalNrmse2D = openCLResources.getContext().createProgram(programStringGetGlobalNrmse2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalNrmse2D = programGetGlobalNrmse2D.createCLKernel("kernelGetGlobalNrmse2D");

        int argn = 0;
        kernelGetGlobalNrmse2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalNrmse2D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalNrmse2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalNrmse2D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalNrmse2D.setArg(argn++, clRepetitionMap);

        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth/64 + ((imageWidth%64==0)?0:1);
        int nYBlocks = imageHeight/64 + ((imageHeight%64==0)?0:1);
        int totalBlocks = nXBlocks * nYBlocks; // Total number of blocks

        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, imageHeight-nYB*64);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, imageWidth-nXB*64);
                float progressPercentage = ((nYB*nXBlocks+nXB)/(float)totalBlocks)*100;
                showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                openCLResources.getQueue().put2DRangeKernel(kernelGetGlobalNrmse2D, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                openCLResources.getQueue().finish();
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                float similarity = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                float weightSum = clWeightsSumMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                repetitionMap[y*imageWidth+x] = similarity/(weightSum*nPixels+Utils.EPSILON);
                openCLResources.getQueue().finish();
            }
        }

        // Invert NRMSE
        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                int index = y*imageWidth+x;
                if(relevanceMask.getRelevanceMask()[index]>0.0f) {
                    float rmse = repetitionMap[index];
                    if (rmse == 0.0f) { // Special case where RMSE is 0, 1/rmse would be undefined but we want perfect similarity
                        repetitionMap[index] = 1.0f;
                    } else {
                        repetitionMap[index] = 1.0f / rmse;
                    }
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, blockRadiusWidth,
                    blockRadiusHeight, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the absolute difference of standard deviations for a 2D image using OpenCL.
     *
     * This method calculates the global repetition map based on the absolute difference of standard deviations. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage2D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally  relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an OpenCLResources object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalAbsDiffStds2D(Utils.InputImage2D inputImage, CLLocalStatistics localStatistics,
                                                 int blockWidth, int blockHeight, Utils.RelevanceMask relevanceMask,
                                                 float nPixels, boolean normalizeOutput, OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalAbsDiffStds2D = getResourceAsString(CLUtils.class, "kernelGetGlobalAbsDiffStds2D.cl");
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalAbsDiffStds2D = replaceFirst(programStringGetGlobalAbsDiffStds2D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalAbsDiffStds2D = openCLResources.getContext().createProgram(programStringGetGlobalAbsDiffStds2D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalAbsDiffStds2D = programGetGlobalAbsDiffStds2D.createCLKernel("kernelGetGlobalAbsDiffStds2D");

        int argn = 0;
        kernelGetGlobalAbsDiffStds2D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalAbsDiffStds2D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalAbsDiffStds2D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalAbsDiffStds2D.setArg(argn++, clRepetitionMap);

        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth/64 + ((imageWidth%64==0)?0:1);
        int nYBlocks = imageHeight/64 + ((imageHeight%64==0)?0:1);
        int totalBlocks = nXBlocks * nYBlocks; // Total number of blocks

        for (int nYB=0; nYB<nYBlocks; nYB++) {
            int yWorkSize = min(64, imageHeight-nYB*64);
            for (int nXB=0; nXB<nXBlocks; nXB++) {
                int xWorkSize = min(64, imageWidth-nXB*64);
                float progressPercentage = ((nYB*nXBlocks+nXB)/(float)totalBlocks)*100;
                showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                openCLResources.getQueue().put2DRangeKernel(kernelGetGlobalAbsDiffStds2D, nXB * 64, nYB * 64, xWorkSize, yWorkSize, 0, 0);
                openCLResources.getQueue().finish();
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                float similarity = clRepetitionMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                float weightSum = clWeightsSumMap.getBuffer().get(y*imageWidth+x);
                openCLResources.getQueue().finish();

                repetitionMap[y*imageWidth+x] = similarity/(weightSum*nPixels+Utils.EPSILON);
                openCLResources.getQueue().finish();
            }
        }

        // Invert repetition map
        for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
            for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                int index = y*imageWidth+x;
                if(relevanceMask.getRelevanceMask()[index]>0.0f) {
                    float srs = repetitionMap[index];
                    if (srs == 0.0f) { // Special case where SRS is 0, 1/SRS would be undefined but we want perfect similarity
                        repetitionMap[index] = 1.0f;
                    } else {
                        repetitionMap[index] = 1.0f / srs;
                    }
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask2D(repetitionMap, imageWidth, imageHeight, relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage2D(repetitionMap, imageWidth, imageHeight, blockRadiusWidth,
                    blockRadiusHeight, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Calculate a global repetition map for 2D data, based on the selected metric.
     *
     * @param metric                The selected metric for repetition calculation.
     * @param inputImage            A {@link Utils.InputImage2D}.
     * @param blockWidth            The width of the block used for the analysis.
     * @param blockHeight           The height of the block used for the analysis.
     * @param blockSize             The size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse.
     * @param localStatistics       A {@link CLLocalStatistics} object.
     * @param relevanceMask         A {@link Utils.RelevanceMask} object
     * @param nPixels               The number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput       A boolean to either normalize the output or not.
     *
     * @return                      The calculated repetition map.
     */
    public static float[] calculateGlobalRepetitionMap2D(String metric, Utils.InputImage2D inputImage, int blockWidth,
                                                         int blockHeight, int blockSize,
                                                         CLLocalStatistics localStatistics,
                                                         Utils.RelevanceMask relevanceMask, float nPixels,
                                                         boolean normalizeOutput, OpenCLResources openCLResources)
    {

        if(metric.equals("Pearson's R")) {
            return CLUtils.getGlobalPearson2D(inputImage, localStatistics, blockWidth, blockHeight, blockSize,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("Cosine similarity")) {
            return CLUtils.getGlobalCosineSimilarity2D(inputImage, localStatistics, blockWidth, blockHeight,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("SSIM")){
            return CLUtils.getGlobalSsim2D(inputImage, localStatistics, blockWidth, blockHeight, blockSize,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("NRMSE")){
            return CLUtils.getGlobalNrmse2D(inputImage, localStatistics, blockWidth, blockHeight, blockSize,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("Abs. diff. of StdDevs")){
            return CLUtils.getGlobalAbsDiffStds2D(inputImage, localStatistics, blockWidth, blockHeight,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else{
            return null;
        }
    }


    // ------------------------------------------ //
    // ---- METHODS FOR GLOBAL REPETITION 3D ---- //
    // ------------------------------------------ //


    /**
     * Computes the Global Repetition Map based on Pearson correlations for a 3D image using OpenCL.
     *
     * This method calculates the global repetition map based on Pearson correlations. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage3D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockDepth        the depth of the block used for the analysis
     * @param blockSize         the size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalPearson3D(Utils.InputImage3D inputImage, CLLocalStatistics localStatistics,
                                             int blockWidth, int blockHeight, int blockDepth, int blockSize,
                                             Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                             OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalPearson3D = getResourceAsString(CLUtils.class, "kernelGetGlobalPearson3D.cl");
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$DEPTH$", "" + imageDepth);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalPearson3D = replaceFirst(programStringGetGlobalPearson3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalPearson3D = openCLResources.getContext().createProgram(programStringGetGlobalPearson3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalPearson3D = programGetGlobalPearson3D.createCLKernel("kernelGetGlobalPearson3D");

        int argn = 0;
        kernelGetGlobalPearson3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalPearson3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalPearson3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalPearson3D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalPearson3D.setArg(argn++, clRepetitionMap);


        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth / 64 + ((imageWidth % 64 == 0) ? 0 : 1);
        int nYBlocks = imageHeight / 64 + ((imageHeight % 64 == 0) ? 0 : 1);
        int nZBlocks = imageDepth / blockDepth + ((imageDepth % blockDepth == 0) ? 0 : 1);
        int totalBlocks = nXBlocks * nYBlocks * nZBlocks;

        for (int nZB = 0; nZB < nZBlocks; nZB++) {
            int zWorkSize = min(blockDepth, imageDepth - nZB * blockDepth);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, imageHeight - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, imageWidth - nXB * 64);
                    float progressPercentage = ((nZB * nYBlocks * nXBlocks) + (nYB * nXBlocks) + nXB) / (float) totalBlocks * 100;
                    showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                    openCLResources.getQueue().put3DRangeKernel(kernelGetGlobalPearson3D,
                            nXB * 64, nYB * 64, nZB * blockDepth,
                            xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    openCLResources.getQueue().finish();
                }
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float similarity = clRepetitionMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    float weightSum = clWeightsSumMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    repetitionMap[index] = similarity / (weightSum * nPixels + Utils.EPSILON);
                    openCLResources.getQueue().finish();
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Cosine Similarity metric for a 3D image using OpenCL.
     *
     * This method calculates the global repetition map based on the Cosine Similarity metric. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage3D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockDepth        the depth of the block used for the analysis
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalCosineSimilarity3D(Utils.InputImage3D inputImage, CLLocalStatistics localStatistics,
                                                      int blockWidth, int blockHeight, int blockDepth,
                                                      Utils.RelevanceMask relevanceMask, float nPixels,
                                                      boolean normalizeOutput, OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalCosineSimilarity3D = getResourceAsString(CLUtils.class, "kernelGetGlobalCosineSimilarity3D.cl");
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$DEPTH$", "" + imageDepth);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalCosineSimilarity3D = replaceFirst(programStringGetGlobalCosineSimilarity3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalCosineSimilarity3D = openCLResources.getContext().createProgram(programStringGetGlobalCosineSimilarity3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalCosineSimilarity3D = programGetGlobalCosineSimilarity3D.createCLKernel("kernelGetGlobalCosineSimilarity3D");

        int argn = 0;
        kernelGetGlobalCosineSimilarity3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalCosineSimilarity3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalCosineSimilarity3D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalCosineSimilarity3D.setArg(argn++, clRepetitionMap);

        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth / 64 + ((imageWidth % 64 == 0) ? 0 : 1);
        int nYBlocks = imageHeight / 64 + ((imageHeight % 64 == 0) ? 0 : 1);
        int nZBlocks = imageDepth / blockDepth + ((imageDepth % blockDepth == 0) ? 0 : 1);
        int totalBlocks = nXBlocks * nYBlocks * nZBlocks;

        for (int nZB = 0; nZB < nZBlocks; nZB++) {
            int zWorkSize = min(blockDepth, imageDepth - nZB * blockDepth);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, imageHeight - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, imageWidth - nXB * 64);
                    float progressPercentage = ((nZB * nYBlocks * nXBlocks) + (nYB * nXBlocks) + nXB) / (float) totalBlocks * 100;
                    showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                    openCLResources.getQueue().put3DRangeKernel(kernelGetGlobalCosineSimilarity3D,
                            nXB * 64, nYB * 64, nZB * blockDepth,
                            xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    openCLResources.getQueue().finish();
                }
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float similarity = clRepetitionMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    float weightSum = clWeightsSumMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    repetitionMap[index] = similarity / (weightSum * nPixels + Utils.EPSILON);
                    openCLResources.getQueue().finish();
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Structural Similarity Index metric (SSIM) for a 3D image using OpenCL.
     *
     * This method calculates the global repetition map based on the Structural Similarity Index metric (SSIM). The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage3D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockDepth        the depth of the block used for the analysis
     * @param blockSize         the size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalSsim3D(Utils.InputImage3D inputImage, CLLocalStatistics localStatistics,
                                             int blockWidth, int blockHeight, int blockDepth, int blockSize,
                                             Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                             OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalSsim3D = getResourceAsString(CLUtils.class, "kernelGetGlobalSsim3D.cl");
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$DEPTH$", "" + imageDepth);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalSsim3D = replaceFirst(programStringGetGlobalSsim3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalSsim3D = openCLResources.getContext().createProgram(programStringGetGlobalSsim3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalSsim3D = programGetGlobalSsim3D.createCLKernel("kernelGetGlobalSsim3D");

        int argn = 0;
        kernelGetGlobalSsim3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalSsim3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalSsim3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalSsim3D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalSsim3D.setArg(argn++, clRepetitionMap);


        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth / 64 + ((imageWidth % 64 == 0) ? 0 : 1);
        int nYBlocks = imageHeight / 64 + ((imageHeight % 64 == 0) ? 0 : 1);
        int nZBlocks = imageDepth / blockDepth + ((imageDepth % blockDepth == 0) ? 0 : 1);
        int totalBlocks = nXBlocks * nYBlocks * nZBlocks;

        for (int nZB = 0; nZB < nZBlocks; nZB++) {
            int zWorkSize = min(blockDepth, imageDepth - nZB * blockDepth);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, imageHeight - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, imageWidth - nXB * 64);
                    float progressPercentage = ((nZB * nYBlocks * nXBlocks) + (nYB * nXBlocks) + nXB) / (float) totalBlocks * 100;
                    showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                    openCLResources.getQueue().put3DRangeKernel(kernelGetGlobalSsim3D,
                            nXB * 64, nYB * 64, nZB * blockDepth,
                            xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    openCLResources.getQueue().finish();
                }
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float similarity = clRepetitionMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    float weightSum = clWeightsSumMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    repetitionMap[index] = similarity / (weightSum * nPixels + Utils.EPSILON);
                    openCLResources.getQueue().finish();
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the Normalized Root Mean Squared Error (NRMSE) for a 3D image using OpenCL.
     *
     * This method calculates the global repetition map based on the Normalized Root Mean Squared Error (NRMSE). The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage3D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockDepth        the depth of the block used for the analysis
     * @param blockSize         the size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalNrmse3D(Utils.InputImage3D inputImage, CLLocalStatistics localStatistics,
                                             int blockWidth, int blockHeight, int blockDepth, int blockSize,
                                             Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                             OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalNrmse3D = getResourceAsString(CLUtils.class, "kernelGetGlobalNrmse3D.cl");
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$DEPTH$", "" + imageDepth);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalNrmse3D = replaceFirst(programStringGetGlobalNrmse3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalNrmse3D = openCLResources.getContext().createProgram(programStringGetGlobalNrmse3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalNrmse3D = programGetGlobalNrmse3D.createCLKernel("kernelGetGlobalNrmse3D");

        int argn = 0;
        kernelGetGlobalNrmse3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalNrmse3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalNrmse3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalNrmse3D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalNrmse3D.setArg(argn++, clRepetitionMap);


        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth / 64 + ((imageWidth % 64 == 0) ? 0 : 1);
        int nYBlocks = imageHeight / 64 + ((imageHeight % 64 == 0) ? 0 : 1);
        int nZBlocks = imageDepth / blockDepth + ((imageDepth % blockDepth == 0) ? 0 : 1);
        int totalBlocks = nXBlocks * nYBlocks * nZBlocks;

        for (int nZB = 0; nZB < nZBlocks; nZB++) {
            int zWorkSize = min(blockDepth, imageDepth - nZB * blockDepth);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, imageHeight - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, imageWidth - nXB * 64);
                    float progressPercentage = ((nZB * nYBlocks * nXBlocks) + (nYB * nXBlocks) + nXB) / (float) totalBlocks * 100;
                    showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                    openCLResources.getQueue().put3DRangeKernel(kernelGetGlobalNrmse3D,
                            nXB * 64, nYB * 64, nZB * blockDepth,
                            xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    openCLResources.getQueue().finish();
                }
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float similarity = clRepetitionMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    float weightSum = clWeightsSumMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    repetitionMap[index] = similarity / (weightSum * nPixels + Utils.EPSILON);
                    openCLResources.getQueue().finish();
                }
            }
        }

        // Invert NRMSE
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    if (relevanceMask.getRelevanceMask()[index] > 0.0f) {
                        float rmse = repetitionMap[index];
                        if (rmse == 0.0f) { // Special case where RMSE is 0, 1/rmse would be undefined but we want perfect similarity
                            repetitionMap[index] = 1.0f;
                        } else {
                            repetitionMap[index] = 1.0f / rmse;
                        }
                    }
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Computes the Global Repetition Map based on the absolute difference of standard deviations for a 3D image using OpenCL.
     *
     * This method calculates the global repetition map based on the absolute difference of standard deviations. The result is stored in a global
     * repetition map, indicating the relative repetition of each structural element across the image.
     * Optionally, a relevance mask can be applied to filter the results based on local standard deviations and a
     * relevance constant.
     *
     * @param inputImage        a {@link Utils.InputImage3D} object
     * @param localStatistics   a {@link CLLocalStatistics} object
     * @param blockWidth        the width of the block used for the analysis
     * @param blockHeight       the height of the block used for the analysis
     * @param blockDepth        the depth of the block used for the analysis
     * @param blockSize         the size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse
     * @param relevanceMask     a {@link Utils.RelevanceMask} object
     * @param nPixels           the number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput   a boolean to either normalize the output or not
     * @param openCLResources   an {@link OpenCLResources} object
     * @return a float array representing the global repetition map, indicating the degree of repetition of each
     * structural element across the entire image.
     */
    public static float[] getGlobalAbsDiffStds3D(Utils.InputImage3D inputImage, CLLocalStatistics localStatistics,
                                           int blockWidth, int blockHeight, int blockDepth, int blockSize,
                                           Utils.RelevanceMask relevanceMask, float nPixels, boolean normalizeOutput,
                                           OpenCLResources openCLResources)
    {
        IJ.log("Calculating Structural Repetition Scores...");

        // Cache variables
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        int imageDepth = inputImage.getDepth();
        int imageSize = inputImage.getSize();
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        // Create and fill OpenCL buffers
        float[] repetitionMap = new float[imageSize];
        CLBuffer<FloatBuffer> clRepetitionMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, repetitionMap);

        float[] weightsSumMap = new float[imageSize];
        CLBuffer<FloatBuffer> clWeightsSumMap = CLUtils.createAndFillCLBuffer(openCLResources.getContext(), imageSize,
                READ_WRITE, weightsSumMap);

        // Build OpenCL program
        String programStringGetGlobalAbsDiffStds3D = getResourceAsString(CLUtils.class, "kernelGetGlobalAbsDiffStds3D.cl");
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$WIDTH$", "" + imageWidth);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$HEIGHT$", "" + imageHeight);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$DEPTH$", "" + imageDepth);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$BLOCK_SIZE$", "" + blockSize);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$BRW$", "" + blockRadiusWidth);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$BRH$", "" + blockRadiusHeight);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$BRZ$", "" + blockRadiusDepth);
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$FILTER_PARAM$", "" + relevanceMask.getNoiseMeanVariance());
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$THRESHOLD$", "" + relevanceMask.getRelevanceThreshold());
        programStringGetGlobalAbsDiffStds3D = replaceFirst(programStringGetGlobalAbsDiffStds3D, "$EPSILON$", "" + Utils.EPSILON);
        CLProgram programGetGlobalAbsDiffStds3D = openCLResources.getContext().createProgram(programStringGetGlobalAbsDiffStds3D).build();

        // Create OpenCL kernel and set args
        CLKernel kernelGetGlobalAbsDiffStds3D = programGetGlobalAbsDiffStds3D.createCLKernel("kernelGetGlobalAbsDiffStds3D");

        int argn = 0;
        kernelGetGlobalAbsDiffStds3D.setArg(argn++, localStatistics.getCLImageArray());
        kernelGetGlobalAbsDiffStds3D.setArg(argn++, localStatistics.getCLLocalMeans());
        kernelGetGlobalAbsDiffStds3D.setArg(argn++, localStatistics.getCLLocalStds());
        kernelGetGlobalAbsDiffStds3D.setArg(argn++, clWeightsSumMap);
        kernelGetGlobalAbsDiffStds3D.setArg(argn++, clRepetitionMap);


        // Calculate weighted mean Pearson's map
        int nXBlocks = imageWidth / 64 + ((imageWidth % 64 == 0) ? 0 : 1);
        int nYBlocks = imageHeight / 64 + ((imageHeight % 64 == 0) ? 0 : 1);
        int nZBlocks = imageDepth / blockDepth + ((imageDepth % blockDepth == 0) ? 0 : 1);
        int totalBlocks = nXBlocks * nYBlocks * nZBlocks;

        for (int nZB = 0; nZB < nZBlocks; nZB++) {
            int zWorkSize = min(blockDepth, imageDepth - nZB * blockDepth);
            for (int nYB = 0; nYB < nYBlocks; nYB++) {
                int yWorkSize = min(64, imageHeight - nYB * 64);
                for (int nXB = 0; nXB < nXBlocks; nXB++) {
                    int xWorkSize = min(64, imageWidth - nXB * 64);
                    float progressPercentage = ((nZB * nYBlocks * nXBlocks) + (nYB * nXBlocks) + nXB) / (float) totalBlocks * 100;
                    showStatus(String.format("Calculating global repetition... %d%%", Math.round(progressPercentage)));
                    openCLResources.getQueue().put3DRangeKernel(kernelGetGlobalAbsDiffStds3D,
                            nXB * 64, nYB * 64, nZB * blockDepth,
                            xWorkSize, yWorkSize, zWorkSize, 0, 0, 0);
                    openCLResources.getQueue().finish();
                }
            }
        }
        showStatus("Calculating global repetition... 100%");

        // Read the repetition map back from the device and calculate the weighted average
        openCLResources.getQueue().putReadBuffer(clRepetitionMap, true);
        openCLResources.getQueue().putReadBuffer(clWeightsSumMap, true);

        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++){
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {

                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    float similarity = clRepetitionMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    float weightSum = clWeightsSumMap.getBuffer().get(index);
                    openCLResources.getQueue().finish();

                    repetitionMap[index] = similarity / (weightSum * nPixels + Utils.EPSILON);
                    openCLResources.getQueue().finish();
                }
            }
        }

        // Invert repetition map
        for(int z=blockRadiusDepth; z<imageDepth-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<imageHeight-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<imageWidth-blockRadiusWidth; x++) {
                    int index = imageWidth*imageHeight*z+y*imageWidth+x;
                    if (relevanceMask.getRelevanceMask()[index] > 0.0f) {
                        float diffStd = repetitionMap[index];
                        if (diffStd == 0.0f) { // Special case where diffStd is 0, 1/diffStd would be undefined but we want perfect similarity
                            repetitionMap[index] = 1.0f;
                        } else {
                            repetitionMap[index] = 1.0f / diffStd;
                        }
                    }
                }
            }
        }

        // Apply relevance mask
        if(relevanceMask.getRelevanceConstant()>0.0f) {
            repetitionMap = Utils.applyMask3D(repetitionMap, imageWidth, imageHeight, imageDepth,
                    relevanceMask.getRelevanceMask());
        }

        // Normalize repetition map (avoiding masked pixels)
        if(normalizeOutput) {
            repetitionMap = Utils.normalizeImage3D(repetitionMap, imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, relevanceMask.getRelevanceMask());
        }

        // Release memory
        openCLResources.getContext().release();

        return repetitionMap;
    }


    /**
     * Calculate a global repetition map for 3D data, based on the selected metric.
     *
     * @param metric                The selected metric for repetition calculation.
     * @param inputImage            A {@link Utils.InputImage2D}.
     * @param blockWidth            The width of the block used for the analysis.
     * @param blockHeight           The height of the block used for the analysis.
     * @param blockDepth            The depth of the block used for the analysis.
     * @param blockSize             The size of the block used for the analysis, after removing pixels outside the inbound circle/ellipse.
     * @param localStatistics       A {@link CLLocalStatistics} object.
     * @param relevanceMask         A {@link Utils.RelevanceMask} object
     * @param nPixels               The number of structurally relevant pixels (i.e., non-masked pixels)
     * @param normalizeOutput       A boolean to either normalize the output or not.
     *
     * @return The calculated repetition map.
     */
    public static float[] calculateGlobalRepetitionMap3D(String metric, Utils.InputImage3D inputImage, int blockWidth,
                                                         int blockHeight, int blockDepth, int blockSize,
                                                         CLLocalStatistics localStatistics,
                                                         Utils.RelevanceMask relevanceMask, float nPixels,
                                                         boolean normalizeOutput, OpenCLResources openCLResources)
    {

        if(metric.equals("Pearson's R")) {
            return CLUtils.getGlobalPearson3D(inputImage, localStatistics, blockWidth, blockHeight, blockDepth,
                    blockSize, relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("Cosine similarity")) {
            return CLUtils.getGlobalCosineSimilarity3D(inputImage, localStatistics, blockWidth, blockHeight, blockDepth,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("SSIM")){
            return CLUtils.getGlobalSsim3D(inputImage, localStatistics, blockWidth, blockHeight, blockDepth, blockSize,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("NRMSE")){
            return CLUtils.getGlobalNrmse3D(inputImage, localStatistics, blockWidth, blockHeight, blockDepth, blockSize,
                    relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else if(metric.equals("Abs. diff. of StdDevs")){
            return CLUtils.getGlobalAbsDiffStds3D(inputImage, localStatistics, blockWidth, blockHeight, blockDepth,
                    blockSize, relevanceMask, nPixels, normalizeOutput, openCLResources);
        }else{
            return null;
        }
    }


}
