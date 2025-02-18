import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.*;
import com.jogamp.opencl.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;

import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

public class CLUtilsTest {

    @Nested
    class OpenCLResourcesTest {

        @Test
        public void testOpenCLResourcesConstructor() {
            // Initialize OpenCL context, device, and queue
            CLContext context = null;
            CLDevice device = null;
            CLCommandQueue queue = null;

            try {
                // Create a default OpenCL context
                CLPlatform platform = CLPlatform.listCLPlatforms()[0]; // Get the first platform
                context = CLContext.create(platform);
                device = context.getMaxFlopsDevice(); // Choose the best device
                queue = device.createCommandQueue(); // Create a command queue

                // Create an instance of OpenCLResources
                CLUtils.OpenCLResources resources = new CLUtils.OpenCLResources(context, device, queue);

                // Assertions to verify the properties of the OpenCLResources instance
                assertNotNull(resources, "OpenCLResources instance should not be null.");
                assertNotNull(resources.getContext(), "OpenCL context should not be null.");
                assertNotNull(resources.getDevice(), "OpenCL device should not be null.");
                assertNotNull(resources.getQueue(), "OpenCL command queue should not be null.");
                assertEquals(context, resources.getContext(), "The context should match.");
                assertEquals(device, resources.getDevice(), "The device should match.");
                assertEquals(queue, resources.getQueue(), "The command queue should match.");

            } catch (CLException e) {
                fail("OpenCL initialization failed: " + e.getMessage());
            } finally {
                // Release resources if they were created
                if (queue != null) queue.release();
                if (context != null) context.release();
            }
        }
    }

    @Nested
    class CLLocalStatisticsTest {

        @Test
        public void testCLLocalStatisticsConstructor() {
            // Sample data for local means and standard deviations
            float[] localMeans = {1.0f, 2.0f, 3.0f};
            float[] localStds = {0.1f, 0.2f, 0.3f};

            // Initialize OpenCL context, device, and queue
            CLContext context = null;
            CLDevice device = null;
            CLCommandQueue queue = null;

            CLPlatform platform = CLPlatform.listCLPlatforms()[0]; // Get the first platform
            context = CLContext.create(platform);
            device = context.getMaxFlopsDevice(); // Choose the best device
            queue = device.createCommandQueue(); // Create a command queue

            // Create mock OpenCL buffers (assuming these are valid)
            CLBuffer<FloatBuffer> clImageArray = context.createFloatBuffer(localMeans.length);
            CLBuffer<FloatBuffer> clLocalMeans = context.createFloatBuffer(localMeans.length);
            CLBuffer<FloatBuffer> clLocalStds = context.createFloatBuffer(localMeans.length);

            // Create an instance of CLLocalStatistics
            CLUtils.CLLocalStatistics localStatistics = new CLUtils.CLLocalStatistics(localMeans, localStds, clImageArray, clLocalMeans, clLocalStds);

            // Assertions to verify the properties of the CLLocalStatistics instance
            assertNotNull(localStatistics, "CLLocalStatistics instance should not be null.");
            assertArrayEquals(localMeans, localStatistics.getLocalMeans(), "Local means should match.");
            assertArrayEquals(localStds, localStatistics.getLocalStds(), "Local standard deviations should match.");
            assertEquals(clImageArray, localStatistics.getCLImageArray(), "CL image array should match.");
            assertEquals(clLocalMeans, localStatistics.getCLLocalMeans(), "CL local means buffer should match.");
            assertEquals(clLocalStds, localStatistics.getCLLocalStds(), "CL local standard deviations buffer should match.");
        }
    }


    // ------------------------------------------------------------------ //
    // ---- METHODS FOR OPENCL INITIALISATION AND RESOURCE MANAGEMENT---- //
    // ------------------------------------------------------------------ //

    @Nested
    class GetOpenCLResourcesTest {

        @Test
        public void testGetOpenCLResources() {

            System.out.println("Library path: " + System.getProperty("java.library.path"));


            // Call the method to get OpenCL resources
            CLUtils.OpenCLResources resources = null;
            boolean useDevice = false; // You can change this to true if you want to test user-defined device preferences

            try {
                resources = CLUtils.getOpenCLResources(useDevice);
            } catch (RuntimeException e) {
                fail("OpenCL initialization failed: " + e.getMessage());
            }

            // Assert that the resources are not null
            assertNotNull(resources, "OpenCLResources should not be null.");

            // Assert that the context, device, and queue are initialized
            assertNotNull(resources.getContext(), "OpenCL context should not be null.");
            assertNotNull(resources.getDevice(), "OpenCL device should not be null.");
            assertNotNull(resources.getQueue(), "OpenCL command queue should not be null.");

            // Assert properties of the device or context
            CLDevice device = resources.getDevice();
            assertTrue(device.getMaxComputeUnits() > 0, "Device should have compute units.");
            assertTrue(device.getMaxClockFrequency() > 0, "Device should have a clock frequency.");
        }
    }


    @Test
    public void testCreateAndFillCLBuffer() {
        // Create a CLContext
        CLUtils.OpenCLResources resources = null;
        try {
            resources = CLUtils.getOpenCLResources(false);
        } catch (RuntimeException e) {
            fail("OpenCL initialization failed: " + e.getMessage());
        }

        // Sample data
        int size = 5;
        float[] inputArray = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Create and fill CLBuffer
        CLBuffer<FloatBuffer> clBuffer = CLUtils.createAndFillCLBuffer(resources.getContext(), size, CLMemory.Mem.READ_WRITE, inputArray);

        // Validate the CLBuffer is not null
        assertNotNull(clBuffer, "CLBuffer should not be null.");

        // Validate that the buffer has the expected size
        assertEquals(size, clBuffer.getBuffer().capacity(), "CLBuffer size should match the input size.");

        // Validate the contents of the buffer
        FloatBuffer bufferContent = clBuffer.getBuffer();
        for (int i = 0; i < size; i++) {
            assertEquals(inputArray[i], bufferContent.get(i), "Buffer content at index " + i + " should match the input array.");
        }

        // Cleanup
        clBuffer.release(); // Release the CLBuffer
        resources.getContext().release();
    }


    @Test
    public void testFillBufferWithFloatArray() {
        // Create a CLContext
        CLUtils.OpenCLResources resources = null;
        try {
            resources = CLUtils.getOpenCLResources(false);
        } catch (RuntimeException e) {
            fail("OpenCL initialization failed: " + e.getMessage());
        }

        // Sample data
        int size = 5;
        float[] inputArray = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Create an empty CLBuffer
        CLBuffer<FloatBuffer> clBuffer = resources.getContext().createFloatBuffer(size, CLMemory.Mem.READ_WRITE);

        // Fill the CLBuffer with the input array
        CLUtils.fillBufferWithFloatArray(clBuffer, inputArray);

        // Validate the CLBuffer is not null
        assertNotNull(clBuffer, "CLBuffer should not be null.");

        // Validate that the buffer has the expected size
        assertEquals(size, clBuffer.getBuffer().capacity(), "CLBuffer size should match the input size.");

        // Validate the contents of the buffer
        FloatBuffer bufferContent = clBuffer.getBuffer();
        for (int i = 0; i < size; i++) {
            assertEquals(inputArray[i], bufferContent.get(i), "Buffer content at index " + i + " should match the input array.");
        }

        // Cleanup
        clBuffer.release();
        resources.getContext().release();
    }


    @Test
    public void testGetResourceAsString() {
        // Get the resource as a string
        String resourceContent = CLUtils.getResourceAsString(CLUtils.class, "kernelGetBlockPearson2D.cl");

        // Validate that the content is not null
        assertNotNull(resourceContent, "Resource content should not be null.");

        // Validate the content is as expected (update with your expected content)
        String expectedContent = "//#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" +
                "#define w $WIDTH$\n" +
                "#define h $HEIGHT$\n" +
                "#define block_size $BLOCK_SIZE$\n" +
                "#define bW $BW$\n" +
                "#define bH $BH$\n" +
                "#define bRW $BRW$\n" +
                "#define bRH $BRH$\n" +
                "#define ref_std $BLOCK_STD$\n" +
                "#define EPSILON $EPSILON$\n" +
                "\n" +
                "kernel void kernelGetBlockPearson2D(\n" +
                "    global float* block_pixels,\n" +
                "    global float* ref_pixels,\n" +
                "    global float* local_means,\n" +
                "    global float* local_stds,\n" +
                "    global float* pearson_map\n" +
                "){\n" +
                "\n" +
                "    int gx = get_global_id(0);\n" +
                "    int gy = get_global_id(1);\n" +
                "\n" +
                "    // Bound check (avoids borders dynamically based on block dimensions)\n" +
                "    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){\n" +
                "        pearson_map[gy*w+gx] = 0.0f;\n" +
                "        return;\n" +
                "    }\n" +
                "\n" +
                "\n" +
                "    // --------------------------------------------- //\n" +
                "    // ---- Get mean-subtracted reference block ---- //\n" +
                "    // --------------------------------------------- //\n" +
                "\n" +
                "    __local float ref_block[block_size]; // Make a local copy to avoid slower reads from global memory\n" +
                "\n" +
                "    for(int i=0; i<block_size; i++){\n" +
                "        ref_block[i] = block_pixels[i]; // Block is mean-subtracted in the host Java class\n" +
                "    }\n" +
                "\n" +
                "\n" +
                "    // ------------------------------------- //\n" +
                "    // ---- Get comparison block pixels ---- //\n" +
                "    // ------------------------------------- //\n" +
                "\n" +
                "    float comp_block[block_size] = {0.0f};\n" +
                "    int index = 0;\n" +
                "    for(int j=gy-bRH; j<=gy+bRH; j++){\n" +
                "        for(int i=gx-bRW; i<=gx+bRW; i++){\n" +
                "            float dx = (float)(i-gx);\n" +
                "            float dy = (float)(j-gy);\n" +
                "            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){\n" +
                "                comp_block[index] = ref_pixels[j*w+i];\n" +
                "                index++;\n" +
                "            }\n" +
                "        }\n" +
                "    }\n" +
                "\n" +
                "\n" +
                "    // Mean-subtract comparison block\n" +
                "    float comp_mean = local_means[gy*w+gx];\n" +
                "    for(int i=0; i<block_size; i++){\n" +
                "        comp_block[i] = comp_block[i] - comp_mean;\n" +
                "    }\n" +
                "\n" +
                "\n" +
                "    // ------------------------------- //\n" +
                "    // ---- Calculate covariance ----- //\n" +
                "    // ------------------------------- //\n" +
                "\n" +
                "    float covariance = 0.0f;\n" +
                "    for(int i=0; i<block_size; i++){\n" +
                "        covariance += ref_block[i] * comp_block[i];\n" +
                "    }\n" +
                "    covariance /= (float)(block_size-1);\n" +
                "\n" +
                "\n" +
                "    // ----------------------------------------------------- //\n" +
                "    // ---- Calculate Pearson's correlation coefficient ---- //\n" +
                "    // ----------------------------------------------------- //\n" +
                "\n" +
                "    float comp_std = local_stds[gy*w+gx];\n" +
                "\n" +
                "    if(ref_std == 0.0f && comp_std == 0.0f){\n" +
                "        pearson_map[gy*w+gx] = 1.0f; // Special case when both blocks are flat (correlation would be NaN but we want 1 because textures are the same)\n" +
                "    }else if(ref_std==0.0f || comp_std==0.0f){\n" +
                "        pearson_map[gy*w+gx] = 0.0f; // Special case when only one block is flat, correlation would be NaN but we want 0\n" +
                "    }else{\n" +
                "        pearson_map[gy*w+gx] = (float) fmax(0.0f, (float)(covariance / ((ref_std * comp_std) + EPSILON))); // Truncate anti-correlations\n" +
                "    }\n" +
                "}\n"; // Change this to the expected content
        assertEquals(expectedContent, resourceContent, "Resource content should match the expected content.");
    }


    @Nested
    class InputStreamToStringTest {

        @Test
        public void testInputStreamToString() throws IOException {
            // Sample data for the input stream
            String expectedContent = "This is a test string.";
            ByteArrayInputStream inputStream = new ByteArrayInputStream(expectedContent.getBytes("UTF-8"));

            // Call the method under test
            String actualContent = CLUtils.inputStreamToString(inputStream);

            // Validate that the content matches the expected string
            assertEquals(expectedContent, actualContent, "The converted string should match the expected content.");
        }

        @Test
        public void testInputStreamToStringEmpty() throws IOException {
            // Test with an empty InputStream
            ByteArrayInputStream emptyInputStream = new ByteArrayInputStream(new byte[0]);

            // Call the method under test
            String actualContent = CLUtils.inputStreamToString(emptyInputStream);

            // Validate that the content is an empty string
            assertEquals("", actualContent, "The converted string should be empty for an empty InputStream.");
        }

        @Test
        public void testInputStreamToStringWithSpecialChars() throws IOException {
            // Sample data with special characters
            String expectedContent = "Hello, 😊! This is a test.";
            ByteArrayInputStream inputStream = new ByteArrayInputStream(expectedContent.getBytes("UTF-8"));

            // Call the method under test
            String actualContent = CLUtils.inputStreamToString(inputStream);

            // Validate that the content matches the expected string
            assertEquals(expectedContent, actualContent, "The converted string should match the expected content with special characters.");
        }
    }


    @Nested
    class ReplaceFirstTest {

        @Test
        public void testReplaceFirstNormalCase() {
            String source = "The quick brown fox jumps over the lazy dog.";
            String target = "fox";
            String replacement = "cat";
            String expected = "The quick brown cat jumps over the lazy dog.";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(expected, actual, "The first occurrence of the target should be replaced.");
        }

        @Test
        public void testReplaceFirstNotFound() {
            String source = "The quick brown fox jumps over the lazy dog.";
            String target = "rabbit";  // Not present
            String replacement = "cat";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(source, actual, "If the target is not found, the source should remain unchanged.");
        }

        @Test
        public void testReplaceFirstMultipleOccurrences() {
            String source = "The quick brown fox jumps over the lazy fox.";
            String target = "fox";
            String replacement = "cat";
            String expected = "The quick brown cat jumps over the lazy fox.";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(expected, actual, "Only the first occurrence of the target should be replaced.");
        }

        @Test
        public void testReplaceFirstAtStart() {
            String source = "fox jumps over the lazy dog.";
            String target = "fox";
            String replacement = "cat";
            String expected = "cat jumps over the lazy dog.";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(expected, actual, "If the target is at the start, it should be replaced correctly.");
        }

        @Test
        public void testReplaceFirstAtEnd() {
            String source = "The quick brown fox.";
            String target = "fox";
            String replacement = "cat";
            String expected = "The quick brown cat.";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(expected, actual, "If the target is at the end, it should be replaced correctly.");
        }

        @Test
        public void testReplaceFirstEmptySource() {
            String source = "";
            String target = "fox";
            String replacement = "cat";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(source, actual, "An empty source string should remain unchanged.");
        }

        @Test
        public void testReplaceFirstEmptyTarget() {
            String source = "The quick brown fox.";
            String target = ""; // Empty target
            String replacement = "cat";

            String actual = CLUtils.replaceFirst(source, target, replacement);

            assertEquals(source, actual, "An empty target string should not change the source string.");
        }
    }


    @Test
    public void testRoundUp() {
        assertEquals(24, CLUtils.roundUp(8, 20), "Should round 20 up to the next multiple of 8 (24).");
        assertEquals(12, CLUtils.roundUp(4, 12), "Should return 12 as it is already a multiple of 4.");
        assertEquals(32, CLUtils.roundUp(16, 30), "Should round 30 up to the next multiple of 16 (32).");
        assertEquals(8, CLUtils.roundUp(8, 8), "Should return 8 as it is already a multiple of 8.");
        assertEquals(10, CLUtils.roundUp(10, 5), "Should round 5 up to the next multiple of 10 (10).");
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 2D ---- //
    // ----------------------------------------- //


    @Test
    public void testGetLocalStatistics2D() {
        // Create a CLContext, device, and command queue
        CLUtils.OpenCLResources resources = null;
        try {
            resources = CLUtils.getOpenCLResources(false);
        } catch (RuntimeException e) {
            fail("OpenCL initialization failed: " + e.getMessage());
        }

        // Sample input image (2D array for testing)
        int width = 5;
        int height = 5;
        int size = width*height;
        float[] inputImageData = {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f
        };

        // Create an InputImage2D instance
        Utils.InputImage2D inputImage2D = new Utils.InputImage2D(inputImageData, width, height, size);

        // Define block size and epsilon
        int blockWidth = 3;
        int blockHeight = 3;

        // Call the method to test
        CLUtils.CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(
                resources.getContext(),
                resources.getDevice(),
                resources.getQueue(),
                inputImage2D,
                blockWidth,
                blockHeight,
                Utils.EPSILON
        );

        // Validate local means and local stds (these values should be calculated based on the input image)
        float[] localMeans = localStatistics.getLocalMeans();
        float[] localStds = localStatistics.getLocalStds();

        // For testing, we should define what we expect here.
        float[] expectedLocalMeans = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 7.0f, 8.0f, 9.0f, 0.0f,
                                      0.0f, 12.0f, 13.0f, 14.0f, 0.0f,
                                      0.0f, 17.0f, 18.0f, 19.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        float[] expectedLocalStds = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                     0.0f, 3.6055512f, 3.6055512f, 3.6055512f, 0.0f,
                                     0.0f, 3.6055512f, 3.6055512f, 3.6055512f, 0.0f,
                                     0.0f, 3.6055512f, 3.6055512f, 3.6055512f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // Validate the output
        assertNotNull(localMeans, "Local means should not be null.");
        assertNotNull(localStds, "Local standard deviations should not be null.");

        // Check the size of the results
        assertEquals(width * height, localMeans.length, "Local means array size should match image size.");
        assertEquals(width * height, localStds.length, "Local stds array size should match image size.");

        // Validate the values (use an appropriate precision for floats)
        for (int i = 0; i < localMeans.length; i++) {
            assertEquals(expectedLocalMeans[i], localMeans[i], Utils.EPSILON, "Local mean value at index " + i + " is incorrect.");
            assertEquals(expectedLocalStds[i], localStds[i], Utils.EPSILON, "Local std deviation value at index " + i + " is incorrect.");
        }

        // Cleanup
        resources.getContext().release();
    }


    @Test
    public void testGetBlockPearson2D() {
        // Set up the input image data
        int imageWidth = 10;
        int imageHeight = 10;
        int imageSize = imageWidth * imageHeight;
        float[] imageArray = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        Utils.InputImage2D inputImage2D = new Utils.InputImage2D(imageArray, imageWidth, imageHeight, imageSize);

        // Set up the reference block data
        int blockWidth = 3;
        int blockHeight = 3;
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;
        float[] block = {0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f
        };

        int blockSize = 0;
        for (int j = 0; j < blockHeight; j++) {
            for (int i = 0; i < blockWidth; i++) {
                float dx = (float) (i - blockRadiusWidth);
                float dy = (float) (j - blockRadiusHeight);
                if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x - blockRadiusWidth);
                float dy = (float) (y - blockRadiusHeight);
                if (((dx * dx) / (float) (blockRadiusWidth * blockRadiusWidth)) + ((dy * dy) / (float) (blockRadiusHeight * blockRadiusHeight)) <= 1.0f) {
                    blockArray[index] = block[y*blockWidth+x];
                    index++;
                }
            }
        }

        float blockMean = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockStd += (blockArray[i]-blockMean)*(blockArray[i]-blockMean);
        }
        blockStd /= (float) blockSize - 1.0f;
        blockStd = (float)sqrt(blockStd);

        Utils.ReferenceBlock2D referenceBlock2D = new Utils.ReferenceBlock2D(blockArray, blockWidth, blockHeight, blockRadiusWidth, blockRadiusHeight, blockSize, blockMean, blockStd);

        // Initialize OpenCL
        CLUtils.OpenCLResources clResources = CLUtils.getOpenCLResources(false);

        // Set relevance constant, normalize output flag, and useDevice flag
        float relevanceConstant = 0.0f;
        boolean normalizeOutput = true;
        boolean useDevice = false;

        // Call the method
        float[] expectedResult = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.40824822f, 0.40824822f, 0.40824822f, 0.0f, 1.0f, 0.0f, 0.40824822f, 0.40824822f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.40824822f, 0.40824822f, 0.40824822f, 0.0f, 1.0f, 0.0f, 0.40824822f, 0.40824822f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        float[] result = CLUtils.getBlockPearson2D(inputImage2D, referenceBlock2D, relevanceConstant, normalizeOutput, useDevice);

        // Assert that the result is not null and has the correct size
        assertNotNull(result);
        assertEquals(imageSize, result.length);

        // Check if result is normalized
        for (float value : result) {
            assertTrue(value >= 0.0f && value <= 1.0f);  // Assuming result is normalized
        }

        // Check if result is as expected
        for(int i=0; i<imageSize; i++){
            assertEquals(expectedResult[i], result[i], 0.000001f);
        }
    }





    @Test
    public void testGetBlockSsim2D() {
        // Set up the input image data
        int imageWidth = 10;
        int imageHeight = 10;
        int imageSize = imageWidth * imageHeight;
        float[] imageArray = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        Utils.InputImage2D inputImage2D = new Utils.InputImage2D(imageArray, imageWidth, imageHeight, imageSize);

        // Set up the reference block data
        int blockWidth = 3;
        int blockHeight = 3;
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;
        float[] block = {0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f
        };

        int blockSize = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x-blockRadiusWidth);
                float dy = (float) (y-blockRadiusHeight);
                if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x - blockRadiusWidth);
                float dy = (float) (y - blockRadiusHeight);
                if (((dx * dx) / (float) (blockRadiusWidth * blockRadiusWidth)) + ((dy * dy) / (float) (blockRadiusHeight * blockRadiusHeight)) <= 1.0f) {
                    blockArray[index] = block[y*blockWidth+x];
                    index++;
                }
            }
        }

        float blockMean = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockStd += (blockArray[i]-blockMean)*(blockArray[i]-blockMean);
        }
        blockStd /= (float) blockSize - 1.0f;
        blockStd = (float)sqrt(blockStd);

        Utils.ReferenceBlock2D referenceBlock2D = new Utils.ReferenceBlock2D(blockArray, blockWidth, blockHeight, blockRadiusWidth, blockRadiusHeight, blockSize, blockMean, blockStd);

        // Initialize OpenCL
        CLUtils.OpenCLResources clResources = CLUtils.getOpenCLResources(false);

        // Set relevance constant, normalize output flag, and useDevice flag
        float relevanceConstant = 0.0f;
        boolean normalizeOutput = true;
        boolean useDevice = false;

        // Call the method
        float[] expectedResult = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9999999f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9999999f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.24068691f, 0.24068691f, 0.24068691f, 0.0f, 0.9999999f, 0.0f, 0.24068691f, 0.24068691f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.24068692f, 0.24068692f, 0.24068692f, 0.0f, 0.9999999f, 0.0f, 0.24068692f, 0.24068692f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9999999f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9999999f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9999999f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
                                  };


        float[] result = CLUtils.getBlockSsim2D(inputImage2D, referenceBlock2D, relevanceConstant, normalizeOutput, useDevice);

        // Assert that the result is not null and has the correct size
        assertNotNull(result);
        assertEquals(imageSize, result.length);

        // Check if result is normalized
        for (float value : result) {
            assertTrue(value >= 0.0f && value <= 1.0f);  // Assuming result is normalized
        }

        // Check if result is as expected
        for(int i=0; i<imageSize; i++){
            assertEquals(expectedResult[i], result[i], Utils.EPSILON);
        }
    }


    @Test
    public void testGetBlockNrmse2D() {
        // Set up the input image data
        int imageWidth = 10;
        int imageHeight = 10;
        int imageSize = imageWidth * imageHeight;
        float[] imageArray = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        Utils.InputImage2D inputImage2D = new Utils.InputImage2D(imageArray, imageWidth, imageHeight, imageSize);

        // Set up the reference block data
        int blockWidth = 3;
        int blockHeight = 3;
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;
        float[] block = {0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f
        };

        int blockSize = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x-blockRadiusWidth);
                float dy = (float) (y-blockRadiusHeight);
                if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x - blockRadiusWidth);
                float dy = (float) (y - blockRadiusHeight);
                if (((dx * dx) / (float) (blockRadiusWidth * blockRadiusWidth)) + ((dy * dy) / (float) (blockRadiusHeight * blockRadiusHeight)) <= 1.0f) {
                    blockArray[index] = block[y*blockWidth+x];
                    index++;
                }
            }
        }

        float blockMean = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockStd += (blockArray[i]-blockMean)*(blockArray[i]-blockMean);
        }
        blockStd /= (float) blockSize - 1.0f;
        blockStd = (float)sqrt(blockStd);

        Utils.ReferenceBlock2D referenceBlock2D = new Utils.ReferenceBlock2D(blockArray, blockWidth, blockHeight, blockRadiusWidth, blockRadiusHeight, blockSize, blockMean, blockStd);

        // Initialize OpenCL
        CLUtils.OpenCLResources clResources = CLUtils.getOpenCLResources(false);

        // Set relevance constant, normalize output flag, and useDevice flag
        float relevanceConstant = 0.0f;
        boolean normalizeOutput = true;
        boolean useDevice = false;

        // Call the method
        float[] expectedResult = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.11117814f, 0.0f, 0.11117814f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.11117814f, 0.0f, 0.11117814f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.18945144f, 0.0f, 0.18945158f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 1.3763912E-7f, 1.3763912E-7f, 1.3763912E-7f, 1.3763912E-7f, 0.99999994f, 1.3763912E-7f, 1.3763912E-7f, 1.3763912E-7f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.18945144f, 0.0f, 0.18945144f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.11117814f, 0.0f, 0.11117814f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.11117814f, 0.0f, 0.11117814f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.99999994f, 0.99999994f, 0.99999994f, 0.11117814f, 0.0f, 0.11117814f, 0.99999994f, 0.99999994f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };


        float[] result = CLUtils.getBlockNrmse2D(inputImage2D, referenceBlock2D, relevanceConstant, normalizeOutput, useDevice);
        //System.out.println(Arrays.toString(result));
        //System.out.println(Arrays.toString(expectedResult));

        // Assert that the result is not null and has the correct size
        assertNotNull(result);
        assertEquals(imageSize, result.length);

        // Check if result is normalized
        for (float value : result) {
            assertTrue(value >= 0.0f && value <= 1.0f);  // Assuming result is normalized
        }

        // Check if result is as expected
        for(int i=0; i<imageSize; i++){
            assertEquals(expectedResult[i], result[i], Utils.EPSILON*100.0f);
        }
    }


    @Test
    public void testGetBlockAbsDiffStds2D() {
        // Set up the input image data
        int imageWidth = 10;
        int imageHeight = 10;
        int imageSize = imageWidth * imageHeight;
        float[] imageArray = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        Utils.InputImage2D inputImage2D = new Utils.InputImage2D(imageArray, imageWidth, imageHeight, imageSize);

        // Set up the reference block data
        int blockWidth = 3;
        int blockHeight = 3;
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;
        float[] block = {0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 1.0f, 0.0f};

        int blockSize = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x-blockRadiusWidth);
                float dy = (float) (y-blockRadiusHeight);
                if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x - blockRadiusWidth);
                float dy = (float) (y - blockRadiusHeight);
                if (((dx * dx) / (float) (blockRadiusWidth * blockRadiusWidth)) + ((dy * dy) / (float) (blockRadiusHeight * blockRadiusHeight)) <= 1.0f) {
                    blockArray[index] = block[y*blockWidth+x];
                    index++;
                }
            }
        }

        float blockMean = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockStd += (blockArray[i]-blockMean)*(blockArray[i]-blockMean);
        }
        blockStd /= (float) blockSize - 1.0f;
        blockStd = (float)sqrt(blockStd);

        Utils.ReferenceBlock2D referenceBlock2D = new Utils.ReferenceBlock2D(blockArray, blockWidth, blockHeight,
                blockRadiusWidth, blockRadiusHeight, blockSize, blockMean, blockStd);

        // Initialize OpenCL
        CLUtils.OpenCLResources clResources = CLUtils.getOpenCLResources(false);

        // Set relevance constant, normalize output flag, and useDevice flag
        float relevanceConstant = 0.0f;
        boolean normalizeOutput = true;
        boolean useDevice = false;

        // Call the method
        float[] expectedResult = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.8164966f, 0.9999999f, 0.8164966f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.8164966f, 0.9999999f, 0.8164966f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.8164966f, 0.8164966f, 0.8164966f, 0.99999976f, 0.9999999f, 0.99999976f, 0.8164966f, 0.8164966f, 0.0f,
                                  0.0f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 0.0f, 0.9999999f, 0.9999999f, 0.9999999f, 0.0f,
                                  0.0f, 0.8164966f, 0.8164966f, 0.8164966f, 0.99999976f, 0.9999999f, 0.99999976f, 0.8164966f, 0.8164966f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.8164966f, 0.9999999f, 0.8164966f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.8164966f, 0.9999999f, 0.8164966f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.8164966f, 0.9999999f, 0.8164966f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };

        float[] result = CLUtils.getBlockAbsDiffStds2D(inputImage2D, referenceBlock2D, relevanceConstant,
                normalizeOutput, useDevice);

        // Assert that the result is not null and has the correct size
        assertNotNull(result);
        assertEquals(imageSize, result.length);

        // Check if result is normalized
        for (float value : result) {
            assertTrue(value >= 0.0f && value <= 1.0f);  // Assuming result is normalized
        }

        // Check if result is as expected
        for(int i=0; i<imageSize; i++){
            assertEquals(expectedResult[i], result[i], Utils.EPSILON);
        }
    }

    @Test
    public void testCalculateBlockRepetitionMap2DAllMetrics() {
        // Mock input data for testing
        int imageWidth = 10;
        int imageHeight = 10;
        int imageSize = imageWidth * imageHeight;
        float[] imageArray = {
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        Utils.InputImage2D inputImage = new Utils.InputImage2D(imageArray, imageWidth, imageHeight, imageSize);

        // Mock reference block data
        // Set up the reference block data
        int blockWidth = 3;
        int blockHeight = 3;
        int blockRadiusWidth = blockWidth/2;
        int blockRadiusHeight = blockHeight/2;
        float[] block = {0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f};

        int blockSize = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x-blockRadiusWidth);
                float dy = (float) (y-blockRadiusHeight);
                if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f) {
                    blockSize++;
                }
            }
        }

        float[] blockArray = new float[blockSize];
        int index = 0;
        for (int y=0; y<blockHeight; y++) {
            for (int x=0; x<blockWidth; x++) {
                float dx = (float) (x - blockRadiusWidth);
                float dy = (float) (y - blockRadiusHeight);
                if (((dx * dx) / (float) (blockRadiusWidth * blockRadiusWidth)) + ((dy * dy) / (float) (blockRadiusHeight * blockRadiusHeight)) <= 1.0f) {
                    blockArray[index] = block[y*blockWidth+x];
                    index++;
                }
            }
        }

        float blockMean = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockMean += blockArray[i];
        }
        blockMean /= (float) blockSize;

        float blockStd = 0.0f;
        for(int i=0; i<blockSize; i++){
            blockStd += (blockArray[i]-blockMean)*(blockArray[i]-blockMean);
        }
        blockStd /= (float) blockSize - 1.0f;
        blockStd = (float)sqrt(blockStd);

        Utils.ReferenceBlock2D referenceBlock = new Utils.ReferenceBlock2D(blockArray, blockWidth, blockHeight,
                blockRadiusWidth, blockRadiusHeight, blockSize, blockMean, blockStd);

        float relevanceConstant = 0.0f; // Mock relevance constant
        boolean normalizeOutput = true; // Mock normalization setting
        boolean useDevice = true; // Mock device usage setting

        for (String metric : BlockRepetition2D_.METRICS) {
            // Call the method under test for each metric
            float[] result = CLUtils.calculateBlockRepetitionMap2D(
                    metric, inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);

            // Assert that the result is not null for a valid metric
            assertNotNull(result, "Result should not be null for metric: " + metric);

            // Assert that the result has the expected size (optional, if you know the expected dimensions)
            int expectedLength = inputImage.getWidth() * inputImage.getHeight();
            assertEquals(expectedLength, result.length, "Unexpected result length for metric: " + metric);

            // Optionally, check the range of values if normalization is enabled
            if (normalizeOutput) {
                for (float value : result) {
                    assertTrue(value >= 0.0f && value <= 1.0f,
                            "Value out of normalized range [0.0, 1.0] for metric: " + metric);
                }
            }
        }
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 3D ---- //
    // ----------------------------------------- //

    @Test
    public void testGetLocalStatistics3D() {
        // Create a CLContext, device, and command queue
        CLUtils.OpenCLResources resources = null;
        try {
            resources = CLUtils.getOpenCLResources(false);
        } catch (RuntimeException e) {
            fail("OpenCL initialization failed: " + e.getMessage());
        }

        // Sample input image (5x5x5 3D array for testing)
        int width = 5;
        int height = 5;
        int depth = 5;
        int size = width * height * depth;
        float[] inputImageData = {
                // Slice 1
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f,

                // Slice 2
                26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                46.0f, 47.0f, 48.0f, 49.0f, 50.0f,

                // Slice 3
                51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
                66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f,

                // Slice 4
                76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
                86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f,
                96.0f, 97.0f, 98.0f, 99.0f, 100.0f,

                // Slice 5
                101.0f, 102.0f, 103.0f, 104.0f, 105.0f,
                106.0f, 107.0f, 108.0f, 109.0f, 110.0f,
                111.0f, 112.0f, 113.0f, 114.0f, 115.0f,
                116.0f, 117.0f, 118.0f, 119.0f, 120.0f,
                121.0f, 122.0f, 123.0f, 124.0f, 125.0f
        };

        // Create an InputImage3D instance
        ImageStack mockIms = new ImageStack(width, height, depth);
        for(int z=0; z<depth; z++){
            FloatProcessor mockFp = new FloatProcessor(width, height);
            for(int y=0; y<height; y++){
                for(int x=0; x<width; x++){
                    mockFp.setf(x, y, inputImageData[width*height*z+y*width+x]);
                }
            }
            mockIms.setProcessor(mockFp, z+1);
        }

        ImagePlus mockIp = new ImagePlus("test_input", mockIms);
        Calibration calibration = mockIp.getCalibration();
        Utils.InputImage3D inputImage3D = new Utils.InputImage3D(inputImageData, width, height, depth, size, calibration,1, 10, 100);

        // Define block size and epsilon
        int blockRadiusWidth = 1; //
        int blockRadiusHeight = 1;
        int blockRadiusDepth = 1;

        // Call the method to test
        CLUtils.CLLocalStatistics localStatistics = CLUtils.getLocalStatistics3D(
                resources,
                inputImage3D,
                3, // blockSize
                blockRadiusWidth,
                blockRadiusHeight,
                blockRadiusDepth,
                Utils.EPSILON
        );

        // Validate local means and local stds (these values should be calculated based on the input image)
        float[] localMeans = localStatistics.getLocalMeans();
        float[] localStds = localStatistics.getLocalStds();

        // Define expected values for local means and standard deviations (manually computed for the test data)
        float[] expectedLocalMeans = {
                // Slice 1
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 2
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 21.666666f, 22.666666f, 23.666666f, 0.0f,
                0.0f, 26.666666f, 27.666666f, 28.666666f, 0.0f,
                0.0f, 31.666666f, 32.666666f, 33.666666f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 3
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 46.666666f, 47.666666f, 48.666666f, 0.0f,
                0.0f, 51.666666f, 52.666666f, 53.666666f, 0.0f,
                0.0f, 56.666666f, 57.666666f, 58.666666f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 4
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 71.666666f, 72.666666f, 73.666666f, 0.0f,
                0.0f, 76.666666f, 77.666666f, 78.666666f, 0.0f,
                0.0f, 81.666666f, 82.666666f, 83.666666f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 5
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };

        float[] expectedLocalStds = {
                // Slice 1
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 2
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 3
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 4
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 12.858201f, 12.858201f, 12.858201f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

                // Slice 5
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };

        // Validate the output
        assertNotNull(localMeans, "Local means should not be null.");
        assertNotNull(localStds, "Local standard deviations should not be null.");

        // Check the size of the results
        assertEquals(size, localMeans.length, "Local means array size should match image size.");
        assertEquals(size, localStds.length, "Local stds array size should match image size.");

        // Validate the values (use an appropriate precision for floats)
        for (int i = 0; i < size; i++) {
            assertEquals(expectedLocalMeans[i], localMeans[i], Utils.EPSILON*100.0f, "Local mean value at index " + i + " is incorrect.");
            assertEquals(expectedLocalStds[i], localStds[i], Utils.EPSILON, "Local std deviation value at index " + i + " is incorrect.");
        }

        // Cleanup
        resources.getContext().release();
    }

    @Test
    public void testGetBlockPearson3D() {
        // Set up the input image data (3D image)
        int imageWidth = 10;
        int imageHeight = 10;
        int imageDepth = 5;
        int imageSize = imageWidth * imageHeight * imageDepth;
        float[] imageArray = new float[imageSize];

        // Initialize the 3D image (example with some patterns)
        for (int z = 0; z < imageDepth; z++) {
            for (int y = 0; y < imageHeight; y++) {
                for (int x = 0; x < imageWidth; x++) {
                    int index = imageHeight*imageWidth*z +y*imageWidth+x;
                    if ((x == 5 && y == 5) || (x == 6 && y == 5) || (x == 5 && y == 6) || (x == 6 && y == 6)) {
                        imageArray[index] = 1.0f;
                    } else {
                        imageArray[index] = 0.0f;
                    }
                }
            }
        }


        // Create an InputImage3D instance
        ImageStack mockIms = new ImageStack(imageWidth, imageHeight, imageDepth);
        for(int z=0; z<imageDepth; z++){
            FloatProcessor mockFp = new FloatProcessor(imageWidth, imageHeight);
            for(int y=0; y<imageHeight; y++){
                for(int x=0; x<imageWidth; x++){
                    mockFp.setf(x, y, imageArray[imageWidth*imageHeight*z+y*imageWidth+x]);
                }
            }
            mockIms.setProcessor(mockFp, z+1);
        }

        ImagePlus mockIp = new ImagePlus("test_input", mockIms);
        Calibration calibration = mockIp.getCalibration();
        Utils.InputImage3D inputImage3D = new Utils.InputImage3D(imageArray, imageWidth, imageHeight, imageDepth, imageSize, calibration, 1, 10, 100);

        // Set up the reference block data (3D block)
        int blockWidth = 3;
        int blockHeight = 3;
        int blockDepth = 3;
        int blockRadiusWidth = blockWidth / 2;
        int blockRadiusHeight = blockHeight / 2;
        int blockRadiusDepth = blockDepth / 2;

        float[] block = {
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,

                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,

                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 1.0f, 0.0f
        };

        int blockSize = blockWidth * blockHeight * blockDepth;

        float blockMean = 0.0f;
        for (int i = 0; i < blockSize; i++) {
            blockMean += block[i];
        }
        blockMean /= blockSize;

        float blockStd = 0.0f;
        for (int i = 0; i < blockSize; i++) {
            blockStd += (block[i] - blockMean) * (block[i] - blockMean);
        }
        blockStd /= blockSize - 1;
        blockStd = (float) Math.sqrt(blockStd);

        Utils.ReferenceBlock3D referenceBlock3D = new Utils.ReferenceBlock3D(block, blockWidth, blockHeight, blockDepth, blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, blockSize, blockMean, blockStd);

        // Initialize OpenCL
        CLUtils.OpenCLResources clResources = CLUtils.getOpenCLResources(false);

        // Set relevance constant, normalize output flag, and useDevice flag
        float relevanceConstant = 0.0f;
        boolean normalizeOutput = true;
        boolean useDevice = false;

        // Call the method
        float[] expectedResult = new float[imageWidth*imageHeight*imageDepth];

        float[] result = CLUtils.getBlockPearson3D(inputImage3D, referenceBlock3D, relevanceConstant, normalizeOutput, useDevice);

        // Assert that the result is not null and has the correct size
        assertNotNull(result);
        assertEquals(imageSize, result.length);

        // Check if result is normalized
        for (float value : result) {
            assertTrue(value >= 0.0f && value <= 1.0f);  // Assuming result is normalized
        }

        // Check if result is as expected
        for (int i = 0; i < imageSize; i++) {
            assertEquals(expectedResult[i], result[i], 0.000001f);
        }
    }

}
