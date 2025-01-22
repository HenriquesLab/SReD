import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.junit.jupiter.api.Assertions.*;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Random;

public class UtilsTest {

    @Test
    public void RelevanceMaskTest() {
        // Test data for relevance mask and parameters
        float[] relevanceMask = {1.0f, 0.0f, 1.0f, 0.0f};  // Sample mask values
        float relevanceConstant = 0.75f;
        float threshold = 0.5f;
        float noiseMeanVariance = 0.2f;

        // Create a RelevanceMask object using the constructor
        Utils.RelevanceMask relevanceMaskObject = new Utils.RelevanceMask(relevanceMask, relevanceConstant, threshold, noiseMeanVariance);

        // Verify the relevance mask
        assertArrayEquals(relevanceMask, relevanceMaskObject.getRelevanceMask());

        // Verify the relevance constant
        assertEquals(relevanceConstant, relevanceMaskObject.getRelevanceConstant());

        // Verify the relevance threshold
        assertEquals(threshold, relevanceMaskObject.getRelevanceThreshold());

        // Verify the noise mean variance
        assertEquals(noiseMeanVariance, relevanceMaskObject.getNoiseMeanVariance());
    }

    @Nested
    class ReferenceBlock2DTest {

        private Utils.ReferenceBlock2D referenceBlock;

        @BeforeEach
        public void setUp() {
            // Initialize ReferenceBlock2D with sample values for testing
            float[] pixels = new float[] {1.0f, 2.0f, 3.0f,
                                          1.0f, 2.0f, 3.0f,
                                          1.0f, 2.0f, 3.0f};
            int width = 3;
            int height = 3;
            int radiusWidth = 1;
            int radiusHeight = 1;
            int size = 9;
            float mean = 2.0f;
            float std = 0.816f;
            referenceBlock = new Utils.ReferenceBlock2D(pixels, width, height, radiusWidth, radiusHeight, size, mean, std);
        }

        @Test
        public void testGetPixels() {
            assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f,
                                           1.0f, 2.0f, 3.0f,
                                           1.0f, 2.0f, 3.0f}, referenceBlock.getPixels());
        }

        @Test
        public void testGetWidth() {
            assertEquals(3, referenceBlock.getWidth());
        }

        @Test
        public void testGetHeight() {
            assertEquals(3, referenceBlock.getHeight());
        }

        @Test
        public void testGetRadiusWidth() {
            assertEquals(1, referenceBlock.getRadiusWidth());
        }

        @Test
        public void testGetRadiusHeight() {
            assertEquals(1, referenceBlock.getRadiusHeight());
        }

        @Test
        public void testGetSize() {
            assertEquals(9, referenceBlock.getSize());
        }

        @Test
        public void testGetMean() {
            assertEquals(2.0f, referenceBlock.getMean());
        }

        @Test
        public void testGetStd() {
            assertEquals(0.816f, referenceBlock.getStd(), 0.001);
        }
    }


    @Nested
    class InputImage2DTest {

        private Utils.InputImage2D inputImage;

        @BeforeEach
        public void setUp() {
            // Initialize InputImage2D with sample values for testing
            float[] imageArray = new float[] {0.1f, 0.2f, 0.3f, 0.4f};
            int width = 2;
            int height = 2;
            int size = 4;
            inputImage = new Utils.InputImage2D(imageArray, width, height, size);
        }

        @Test
        public void testGetImageArray() {
            assertArrayEquals(new float[] {0.1f, 0.2f, 0.3f, 0.4f}, inputImage.getImageArray());
        }

        @Test
        public void testGetWidth() {
            assertEquals(2, inputImage.getWidth());
        }

        @Test
        public void testGetHeight() {
            assertEquals(2, inputImage.getHeight());
        }

        @Test
        public void testGetSize() {
            assertEquals(4, inputImage.getSize());
        }
    }


    @Nested
    class ReferenceBlock3DTest {

        private Utils.ReferenceBlock3D referenceBlock;

        @BeforeEach
        public void setUp() {
            // Initialize ReferenceBlock3D with test values
            float[] pixels = new float[] {1.0f, 2.0f, 3.0f, 4.0f};
            int width = 2;
            int height = 2;
            int depth = 1;
            int radiusWidth = 1;
            int radiusHeight = 1;
            int radiusDepth = 0;
            int size = 4;
            float mean = 2.5f;
            float std = 1.118f;

            referenceBlock = new Utils.ReferenceBlock3D(pixels, width, height, depth, radiusWidth, radiusHeight, radiusDepth, size, mean, std);
        }

        @Test
        public void testGetPixels() {
            assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, referenceBlock.getPixels());
        }

        @Test
        public void testGetWidth() {
            assertEquals(2, referenceBlock.getWidth());
        }

        @Test
        public void testGetHeight() {
            assertEquals(2, referenceBlock.getHeight());
        }

        @Test
        public void testGetDepth() {
            assertEquals(1, referenceBlock.getDepth());
        }

        @Test
        public void testGetRadiusWidth() {
            assertEquals(1, referenceBlock.getRadiusWidth());
        }

        @Test
        public void testGetRadiusHeight() {
            assertEquals(1, referenceBlock.getRadiusHeight());
        }

        @Test
        public void testGetRadiusDepth() {
            assertEquals(0, referenceBlock.getRadiusDepth());
        }

        @Test
        public void testGetSize() {
            assertEquals(4, referenceBlock.getSize());
        }

        @Test
        public void testGetMean() {
            assertEquals(2.5f, referenceBlock.getMean(), 0.0001f);
        }

        @Test
        public void testGetStd() {
            assertEquals(1.118f, referenceBlock.getStd(), 0.0001f);
        }
    }


    @Nested
    class InputImage3DTest {

        private Utils.InputImage3D inputImage3D;
        private Calibration calibration;

        @BeforeEach
        public void setUp() {
            // Initialize Calibration object
            calibration = new Calibration();
            calibration.pixelWidth = 0.5;
            calibration.pixelHeight = 0.5;
            calibration.pixelDepth = 1.0;

            // Initialize InputImage3D with test values
            float[] imageArray = new float[] {1.0f, 2.0f, 3.0f, 4.0f};
            int width = 2;
            int height = 2;
            int depth = 1;
            int size = 4;
            float gain = 1.5f;
            float sigma = 0.1f;
            float offset = 0.05f;

            inputImage3D = new Utils.InputImage3D(imageArray, width, height, depth, size, calibration, gain, sigma, offset);
        }

        @Test
        public void testGetImageArray() {
            assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, inputImage3D.getImageArray(), "Image array should match");
        }

        @Test
        public void testGetWidth() {
            assertEquals(2, inputImage3D.getWidth());
        }

        @Test
        public void testGetHeight() {
            assertEquals(2, inputImage3D.getHeight());
        }

        @Test
        public void testGetDepth() {
            assertEquals(1, inputImage3D.getDepth());
        }

        @Test
        public void testGetSize() {
            assertEquals(4, inputImage3D.getSize());
        }

        @Test
        public void testGetCalibration() {
            assertNotNull(inputImage3D.getCalibration());
            assertEquals(0.5, inputImage3D.getCalibration().pixelWidth);
            assertEquals(0.5, inputImage3D.getCalibration().pixelHeight);
            assertEquals(1.0, inputImage3D.getCalibration().pixelDepth);
        }

        @Test
        public void testGetGain() {
            assertEquals(1.5f, inputImage3D.getGain());
        }

        @Test
        public void testGetSigma() {
            assertEquals(0.1f, inputImage3D.getSigma());
        }

        @Test
        public void testGetOffset() {
            assertEquals(0.05f, inputImage3D.getOffset());
        }
    }


    @Nested
    class GetReferenceBlock2DTest {

        private ImagePlus testImage;

        @BeforeEach
        public void setup() {
            // Create a 5x5 float image in memory for testing
            float[] pixels = {
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                    2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                    3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                    4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                    5.0f, 6.0f, 7.0f, 8.0f, 9.0f
            };

            FloatProcessor fp = new FloatProcessor(5, 5, pixels);
            testImage = new ImagePlus("Test Block Image", fp);

            // Assign the image to the WindowManager using a blockID (1 in this case)
            WindowManager.setTempCurrentImage(testImage);  // Temporary block for this test
        }

        @Test
        public void testGetReferenceBlock2DValidBlock() {
            // Valid blockID case (1 in this case, as set in setup)
            Utils.ReferenceBlock2D block = Utils.getReferenceBlock2D(testImage.getID());

            assertNotNull(block, "Block should not be null");

            // Check dimensions
            assertEquals(5, block.getWidth());
            assertEquals(5, block.getHeight());
            assertEquals(2, block.getRadiusWidth());
            assertEquals(2, block.getRadiusHeight());

            // Check block size
            assertTrue(block.getSize() > 0);

            // Check mean and std dev (expected values would depend on the generated block pixels)
            assertEquals(0.5f, block.getMean(), 1e-6);
            assertEquals(0.64549720287323f, block.getStd(), 1e-6);
        }

        @Test
        public void testGetReferenceBlock2DEvenDimensions() {
            // Create a new image with even dimensions (invalid case)
            float[] evenPixels = {
                    1, 2, 3, 4,
                    2, 3, 4, 5,
                    3, 4, 5, 6,
                    4, 5, 6, 7
            };
            FloatProcessor evenFp = new FloatProcessor(4, 4, evenPixels);
            ImagePlus evenImage = new ImagePlus("Even Dimension Image", evenFp);

            // Set the new image as the current image
            WindowManager.setTempCurrentImage(evenImage);

            // Retrieve the block
            Utils.ReferenceBlock2D block = Utils.getReferenceBlock2D(evenImage.getID());

            assertNull(block, "Block should be null for even dimensions");
        }

        @Test
        public void testGetReferenceBlock2DNonExistentBlock() {
            // Simulate a case where block image doesn't exist
            WindowManager.setTempCurrentImage(null);

            Utils.ReferenceBlock2D block = Utils.getReferenceBlock2D(999);  // Non-existent blockID

            assertNull(block, "Block should be null if image not found");
        }
    }


    @Nested
    class GetInputImage2DTest {

        @BeforeEach
        public void setUp() {
            // Clean WindowManager before each test
            WindowManager.closeAllWindows();
        }

        @Test
        public void testGetInputImage2D_NoStabilization_NoNormalization() {
            // Set up image dimensions and data
            int width = 3;
            int height = 3;
            float[] pixels = {
                    0.0f, 0.1f, 0.2f,
                    0.3f, 0.4f, 0.5f,
                    0.6f, 0.7f, 0.8f
            };

            // Create FloatProcessor and ImagePlus
            FloatProcessor fp = new FloatProcessor(width, height, pixels);
            ImagePlus imp = new ImagePlus("Test Image", fp);

            // Add the ImagePlus to the WindowManager
            WindowManager.setTempCurrentImage(imp);
            int imageID = imp.getID();

            // Call the method
            Utils.InputImage2D result = Utils.getInputImage2D(imageID, false, "Simplex", 5000, false);

            // Verify the result
            assertNotNull(result);
            assertArrayEquals(pixels, result.getImageArray());
            assertEquals(width, result.getWidth());
            assertEquals(height, result.getHeight());
            assertEquals(width * height, result.getSize());
        }

        @Test
        public void testGetInputImage2D_WithNormalization() {
            // Set up image dimensions and data
            int width = 2;
            int height = 2;
            float[] pixels = {
                    10.0f, 20.0f,
                    30.0f, 40.0f
            };

            // Expected normalized output
            float[] normalizedPixels = {
                    0.0f, 0.3333f,
                    0.6667f, 1.0f
            };

            // Create FloatProcessor and ImagePlus
            FloatProcessor fp = new FloatProcessor(width, height, pixels);
            ImagePlus imp = new ImagePlus("Test Image", fp);

            // Add the ImagePlus to the WindowManager
            WindowManager.setTempCurrentImage(imp);
            int imageID = imp.getID();

            // Call the method with normalization
            Utils.InputImage2D result = Utils.getInputImage2D(imageID, false, "Simplex", 5000, true);

            // Verify the result
            assertNotNull(result);
            assertEquals(width, result.getWidth());
            assertEquals(height, result.getHeight());
            assertEquals(width * height, result.getSize());

            // Check normalized values with a small delta for floating-point comparison
            for (int i = 0; i < normalizedPixels.length; i++) {
                assertEquals(normalizedPixels[i], result.getImageArray()[i], 0.0001);
            }
        }

        @Test
        public void testGetInputImage2D_ImageNotFound() {
            // Call the method with an invalid imageID
            Utils.InputImage2D result = Utils.getInputImage2D(999, false, "Simplex", 5000, false);

            // Verify that the method returns null when image is not found
            assertNull(result);
        }
    }


    @Nested
    class GetInputImage2DTestOverloaded {

        @Test
        public void testGetInputImage2D_NoStabilization_NoNormalization() {
            // Image setup
            int width = 3;
            int height = 3;
            float[] pixels = {
                    0.0f, 0.1f, 0.2f,
                    0.3f, 0.4f, 0.5f,
                    0.6f, 0.7f, 0.8f
            };

            // Call the method with no stabilization and no normalization
            Utils.InputImage2D result = Utils.getInputImage2D(pixels, width, height, false, 0.0f, 0.0f, 0.0f, false);

            // Verify the result
            assertNotNull(result);
            assertArrayEquals(pixels, result.getImageArray());
            assertEquals(width, result.getWidth());
            assertEquals(height, result.getHeight());
            assertEquals(width * height, result.getSize());
        }

        @Test
        public void testGetInputImage2D_WithNormalization() {
            // Image setup
            int width = 2;
            int height = 2;
            float[] pixels = {
                    10.0f, 20.0f,
                    30.0f, 40.0f
            };

            // Expected normalized output
            float[] normalizedPixels = {
                    0.0f, 0.3333f,
                    0.6667f, 1.0f
            };

            // Call the method with normalization enabled
            Utils.InputImage2D result = Utils.getInputImage2D(pixels, width, height, false, 0.0f, 0.0f, 0.0f, true);

            // Verify the result
            assertNotNull(result);
            assertEquals(width, result.getWidth());
            assertEquals(height, result.getHeight());
            assertEquals(width * height, result.getSize());

            // Check normalized values with a small delta for floating-point comparison
            for (int i = 0; i < normalizedPixels.length; i++) {
                assertEquals(normalizedPixels[i], result.getImageArray()[i], 0.0001);
            }
        }


    }


    @Nested
    class GetMeanNoiseVarTest {

        @Test
        public void testGetMeanNoiseVar2D() {
            // Setup a 100x100 image
            int width = 100;  // Image width
            int height = 100; // Image height
            float[] refPixels = new float[width * height]; // Initialize array for 100x100 image

            // Fill the array with a simple pattern, e.g., a gradient
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    refPixels[y * width + x] = (float) (x + y * width) / (width * height);
                }
            }

            // Calculate total number of pixels
            int wh = width * height; // Total number of pixels

            // Execute
            float result = Utils.getMeanNoiseVar2D(refPixels, width, height, wh);

            // Assertions
            // Check that the result is greater than or equal to zero
            assertTrue(result >= 0.0f, "The noise variance should be non-negative.");

            // You can also add checks for specific expected values based on known input
            // For instance, if you know the expected variance for this input
            // assertEquals(expectedVariance, result, 0.01f); // Adjust tolerance as necessary
        }
    }


    @Nested
    class GetMeanAndVarBlock2DTest {

        @Test
        public void testGetMeanAndVarBlock2D() {
            // Setup a 100x100 image with known pixel values
            int width = 100;
            int height = 100;
            float[] imageArray = new float[width * height];

            // Fill the array with a simple pattern (e.g., x + y)
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    imageArray[y * width + x] = (float) (x + y);
                }
            }

            // Define the block to test (10x10 block starting at (10, 10))
            int xStart = 10;
            int yStart = 10;
            int xEnd = 20;  // Exclusive
            int yEnd = 20;  // Exclusive

            // Execute the method
            float[] meanVar = Utils.getMeanAndVarBlock2D(imageArray, width, xStart, yStart, xEnd, yEnd);

            // Calculate expected mean and variance manually for the block (10x10 block from (10, 10) to (19, 19))
            float sum = 0.0f;
            int totalPixels = (xEnd - xStart) * (yEnd - yStart);  // 10x10 = 100 pixels
            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    sum += (x + y);
                }
            }
            float expectedMean = sum / totalPixels;

            // For variance, you can compute it similarly by iterating over the block and calculating the sum of squared differences
            float varianceSum = 0.0f;
            for (int y = yStart; y < yEnd; y++) {
                for (int x = xStart; x < xEnd; x++) {
                    float value = (x + y);
                    varianceSum += Math.pow(value - expectedMean, 2);
                }
            }
            float expectedVar = varianceSum / totalPixels;

            // Assertions
            assertEquals(expectedMean, meanVar[0], 0.01f, "Mean value should match the expected value.");
            assertEquals(expectedVar, meanVar[1], 0.01f, "Variance value should match the expected value.");
        }
    }


    @Nested
    class ApplyMaskTest {

        @Test
        public void testApplyMask2D() {
            // Setup a 3x3 image with known pixel values
            int width = 3;
            int height = 3;
            float[] imageArray = new float[] {
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f
            };

            // Create a mask with known values (e.g., a mask that zeroes out some values)
            float[] mask = new float[] {
                    1.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f,
                    1.0f, 1.0f, 0.0f
            };

            // Expected result after applying the mask
            float[] expectedResult = new float[] {
                    1.0f, 0.0f, 3.0f,
                    0.0f, 5.0f, 0.0f,
                    7.0f, 8.0f, 0.0f
            };

            // Execute the method
            float[] result = Utils.applyMask2D(imageArray, width, height, mask);

            // Assertions to check if the result matches the expected result
            assertArrayEquals(expectedResult, result);
        }
    }


    @Test
    public void GetRelevanceMaskTest() {
        // Setup a larger 50x50 image with known pixel values
        int width = 50;
        int height = 50;

        // Initialize the imageArray and localStds with some sample data (mock values for illustration)
        float[] imageArray = new float[width * height];
        float[] localStds = new float[width * height];

        // Set a seed for reproducibility
        Random random = new Random(42);  // Fixed seed

        // Fill the arrays with reproducible values
        for (int i = 0; i < width * height; i++) {
            imageArray[i] = (i % 50 == 0) ? 1.0f : 1.0f + random.nextFloat() * 2.0f;  // Values between 1.0f and 3.0f
            localStds[i] = (i % 50 == 0) ? 0.0f : random.nextFloat();                 // Local standard deviations
        }

        // Parameters for the relevance mask
        int blockRadiusWidth = 2;  // Adjusted radius
        int blockRadiusHeight = 2; // Adjusted radius
        float relevanceConstant = 2.0f;

        // Calculate expected noise mean variance based on mock data
        float noiseMeanVariance = 0.25f;  // Assuming pre-calculated based on input

        // Create expected relevance mask (for a large test case, exact expected values may vary)
        float relevanceThreshold = noiseMeanVariance * relevanceConstant;

        // Execute the method
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(imageArray, width, height, blockRadiusWidth, blockRadiusHeight, localStds, relevanceConstant);

        // Assertions to check if the relevance mask is correctly generated
        assertEquals(relevanceConstant, relevanceMask.getRelevanceConstant());
        assertEquals(0.4694742262363434f, relevanceMask.getRelevanceThreshold());
        assertEquals(0.2347371131181717, relevanceMask.getNoiseMeanVariance());

        // Optionally check some values in the mask for correctness (e.g., expected ranges or patterns)
        // Example: Check some central part of the mask
        for (int y = 20; y < 30; y++) {
            for (int x = 20; x < 30; x++) {
                assertTrue(relevanceMask.getRelevanceMask()[y * width + x] == 1.0f || relevanceMask.getRelevanceMask()[y * width + x] == 0.0f);
            }
        }
    }


    @Nested
    class NormalizeImage2DTest {

        @Test
        public void testNormalizeImageWithBinaryMask() {
            // Setup: a small 4x4 image with known pixel values and a valid binary mask
            int width = 4;
            int height = 4;
            int borderWidth = 1;
            int borderHeight = 1;

            float[] imageArray = new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f
            };

            // Binary mask (0.0f or 1.0f only)
            float[] mask = new float[] {
                    1.0f, 0.0f, 1.0f, 0.0f,
                    1.0f, 1.0f, 0.0f, 0.0f,
                    1.0f, 0.0f, 1.0f, 1.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
            };

            // Expected normalized array with the mask applied (min = 1.0, max = 16.0)
            float[] expectedArray = new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 0.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 1.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f
            };

            // Run the normalization with a binary mask
            float[] normalizedArray = Utils.normalizeImage2D(imageArray, width, height, borderWidth, borderHeight, mask);

            // Assertions: Check that the normalized array is as expected
            assertArrayEquals(expectedArray, normalizedArray, 0.0001f);
        }

        @Test
        public void testNormalizeImageWithInvalidMask() {
            // Setup: a small 4x4 image with known pixel values and an invalid mask (non-binary values)
            int width = 4;
            int height = 4;
            int borderWidth = 0;
            int borderHeight = 0;

            float[] imageArray = new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f
            };

            // Invalid mask (contains values other than 0.0f or 1.0f)
            float[] invalidMask = new float[] {
                    1.0f, 0.5f, 1.0f, 0.0f,
                    1.0f, 0.8f, 0.0f, 0.0f,
                    1.0f, 0.0f, 1.0f, 1.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
            };

            // Expect an IllegalArgumentException due to non-binary mask values
            assertThrows(IllegalArgumentException.class, () -> {
                Utils.normalizeImage2D(imageArray, width, height, borderWidth, borderHeight, invalidMask);
            });
        }
    }


    // ----------------------------------------- //
    // ---- METHODS FOR BLOCK REPETITION 3D ---- //
    // ----------------------------------------- //

    @Nested
    class GetReferenceBlock3DTest {

        @Test
        public void testBlockRetrieval() {
            // Create a 3D ImagePlus object (3x3x3 block) and assign it a window ID
            int width = 3;
            int height = 3;
            int depth = 3;
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice with some values
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        fp.setf(x, y, (z + 1) * (x + 1) + y); // Some arbitrary pattern
                    }
                }
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Image", stack);
            imagePlus.show(); // This assigns it an ID in the WindowManager

            // Get the blockID (WindowManager uses 1-based indexing)
            int blockID = imagePlus.getID();

            // Run the method
            Utils.ReferenceBlock3D refBlock = Utils.getReferenceBlock3D(blockID);

            // Assert that the block is not null
            assertNotNull(refBlock);

            // Assert block dimensions
            assertEquals(width, refBlock.getWidth());
            assertEquals(height, refBlock.getHeight());
            assertEquals(depth, refBlock.getDepth());

            // Assert blockSize matches the count of pixels within the ellipsoid
            assertTrue(refBlock.getSize() > 0);

            // Clean up ImageJ windows after the test
            imagePlus.changes = false; // Prevents "Save changes?" prompt
            imagePlus.close();
        }

        @Test
        public void testOddDimensionValidation() {
            // Create a 3D ImagePlus object with even dimensions (should fail)
            int width = 4; // Even
            int height = 3; // Odd
            int depth = 3; // Odd
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Even Width Image", stack);
            imagePlus.show();

            // Run the method and expect an exception due to even dimensions
            int blockID = imagePlus.getID();
            Exception exception = assertThrows(IllegalArgumentException.class, () -> {
                Utils.getReferenceBlock3D(blockID);
            });
            assertEquals("Block dimensions must be odd.", exception.getMessage());

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }

        @Test
        public void testMinimumSlicesValidation() {
            // Create a 3D ImagePlus object with fewer than 3 slices (should fail)
            int width = 3;
            int height = 3;
            int depth = 1; // Less than 3 slices
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Few Slices Image", stack);
            imagePlus.show();

            // Run the method and expect an exception due to fewer than 3 slices
            int blockID = imagePlus.getID();
            Exception exception = assertThrows(IllegalArgumentException.class, () -> {
                Utils.getReferenceBlock3D(blockID);
            });
            assertEquals("Reference block must have at least 3 slices.", exception.getMessage());

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }

        @Test
        public void testNormalizationAndStatistics() {
            // Create a 3x3x3 ImagePlus object for testing normalization and statistics
            int width = 3;
            int height = 3;
            int depth = 3;
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice with a known pattern
            float[][] slices = {
                    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f },
                    { 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f },
                    { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 2.0f, 4.0f, 6.0f, 8.0f }
            };
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                fp.setPixels(slices[z]);
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Image for Stats", stack);
            imagePlus.show();

            // Get the blockID (WindowManager uses 1-based indexing)
            int blockID = imagePlus.getID();

            // Run the method
            Utils.ReferenceBlock3D refBlock = Utils.getReferenceBlock3D(blockID);

            // Validate the statistics of the block
            assertNotNull(refBlock);

            // Ensure that the mean and standard deviation have been computed
            assertTrue(refBlock.getMean() != 0.0f);
            assertTrue(refBlock.getStd() != 0.0f);

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }
    }


    @Nested
    class GetInputImage3DTest {

        @Test
        public void testImageRetrieval() {
            // Create a 3D ImagePlus object (3x3x3 block) and assign it a window ID
            int width = 3;
            int height = 3;
            int depth = 3;
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice with some values
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        fp.setf(x, y, (z + 1) * (x + 1) + y); // Arbitrary values
                    }
                }
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Image", stack);
            imagePlus.show(); // Assigns an ID in WindowManager

            // Get the imageID (WindowManager uses 1-based indexing)
            int imageID = imagePlus.getID();

            // Call the method (without stabilizing noise variance, but normalizing output)
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageID, false, "Quad/Octree", true);

            // Assertions
            assertNotNull(inputImage, "Image should be retrieved successfully.");
            assertEquals(width, inputImage.getWidth(), "Width should match.");
            assertEquals(height, inputImage.getHeight(), "Height should match.");
            assertEquals(depth, inputImage.getDepth(), "Depth should match.");
            assertEquals(width * height * depth, inputImage.getSize(), "Size should match.");

            // Clean up ImageJ windows after the test
            imagePlus.changes = false; // Prevents "Save changes?" prompt
            imagePlus.close();
        }

        @Test
        public void testImageNotFound() {
            // Call method with an invalid image ID
            int invalidImageID = 9999; // Assuming this ID does not exist
            Utils.InputImage3D inputImage = Utils.getInputImage3D(invalidImageID, false, "Quad/Octree", false);

            // Assert that the method returns null when the image is not found
            assertNull(inputImage, "Image should not be found and the method should return null.");
        }

        @Test
        public void testInsufficientSlices() {
            // Create an ImagePlus object with only 2 slices (should fail)
            int width = 3;
            int height = 3;
            int depth = 2; // Less than 3 slices
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Few Slices Image", stack);
            imagePlus.show(); // Assigns an ID in WindowManager

            // Get the imageID
            int imageID = imagePlus.getID();

            // Call the method (should return null due to insufficient slices)
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageID, false, "Quad/Octree", false);

            // Assert that the method returns null when there are fewer than 3 slices
            assertNull(inputImage, "Image must have at least 3 slices, method should return null.");

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }

        @Test
        public void testNormalization() {
            // Create a 3x3x3 ImagePlus object for testing normalization
            int width = 3;
            int height = 3;
            int depth = 3;
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice with a known pattern
            float[][] slices = {
                    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, // Slice 1
                    { 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }, // Slice 2
                    { 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 2.0f, 4.0f, 6.0f, 8.0f } // Slice 3
            };
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                fp.setPixels(slices[z]);
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Image for Normalization", stack);
            imagePlus.show(); // Assigns an ID in WindowManager

            // Get the imageID
            int imageID = imagePlus.getID();

            // Call the method with normalization enabled
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageID, false, "Quad/Octree", true);

            // Assertions
            assertNotNull(inputImage, "Image should be retrieved successfully.");
            float[] normalizedArray = inputImage.getImageArray();
            assertNotNull(normalizedArray, "Array should not be null after normalization.");
            assertEquals(width * height * depth, normalizedArray.length, "Array length should match image size.");

            // Ensure array values are within the normalized range (0.0f - 1.0f)
            for (float value : normalizedArray) {
                assertTrue(value >= 0.0f && value <= 1.0f, "All values should be in the range [0.0, 1.0].");
            }

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }

        @Test
        public void testNoiseVarianceStabilization() {
            // Create a 3x3x3 ImagePlus object for testing noise variance stabilization
            int width = 3;
            int height = 3;
            int depth = 3;
            ImageStack stack = new ImageStack(width, height);

            // Fill each slice with arbitrary values
            for (int z = 0; z < depth; z++) {
                FloatProcessor fp = new FloatProcessor(width, height);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        fp.setf(x, y, (float) (Math.random() * 100)); // Random values
                    }
                }
                stack.addSlice(fp);
            }
            ImagePlus imagePlus = new ImagePlus("Test Image for Noise Variance", stack);
            imagePlus.show(); // Assigns an ID in WindowManager

            // Get the imageID
            int imageID = imagePlus.getID();

            // Call the method with noise variance stabilization enabled
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageID, true, "Simplex", false);

            // Assertions
            assertNotNull(inputImage, "Image should be retrieved successfully.");
            assertEquals(width, inputImage.getWidth(), "Width should match the input image.");
            assertEquals(height, inputImage.getHeight(), "Height should match the input image.");
            assertEquals(depth, inputImage.getDepth(), "Depth should match the input image.");
            assertDoesNotThrow(() -> {
                float gain = inputImage.getGain();
                float sigma = inputImage.getSigma();
                float offset = inputImage.getOffset();
            }, "Gain, sigma, and offset calculations should not throw exceptions.");

            // Clean up ImageJ windows after the test
            imagePlus.changes = false;
            imagePlus.close();
        }
    }


    @Nested
    class GetInputImage3DOverloadTest {

        @Test
        void testGetInputImage3DWithoutNoiseStabilizationOrNormalization() {
            // Setup: 3D image data (2x2x3)
            int width = 2;
            int height = 2;
            int depth = 3;
            float[] imageArray = new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f, // Slice 1
                    5.0f, 6.0f, 7.0f, 8.0f, // Slice 2
                    9.0f, 10.0f, 11.0f, 12.0f // Slice 3
            };

            // Test without stabilization or normalization
            boolean stabiliseNoiseVariance = false;
            boolean normalizeOutput = false;

            // Run the method
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageArray, width, height, depth, stabiliseNoiseVariance, normalizeOutput);

            // Assert that the image was returned correctly
            assertNotNull(inputImage, "Image should be retrieved successfully.");
            assertEquals(width, inputImage.getWidth());
            assertEquals(height, inputImage.getHeight());
            assertEquals(depth, inputImage.getDepth());

            // Check that gain, sigma, and offset are still 0.0f
            assertEquals(0.0f, inputImage.getGain(), 0.0001f);
            assertEquals(0.0f, inputImage.getSigma(), 0.0001f);
            assertEquals(0.0f, inputImage.getOffset(), 0.0001f);
        }

        @Test
        void testGetInputImage3DWithNormalization() {
            // Setup: 3D image data (2x2x3)
            int width = 2;
            int height = 2;
            int depth = 3;
            float[] imageArray = new float[] {
                    1.0f, 2.0f, 3.0f, 4.0f, // Slice 1
                    5.0f, 6.0f, 7.0f, 8.0f, // Slice 2
                    9.0f, 10.0f, 11.0f, 12.0f // Slice 3
            };

            // Test with normalization but no noise stabilization
            boolean stabiliseNoiseVariance = false;
            boolean normalizeOutput = true;

            // Expected normalized array (min = 1.0, max = 12.0)
            float[] expectedArray = new float[] {
                    0.0f, 0.090f, 0.181f, 0.272f,
                    0.363f, 0.454f, 0.545f, 0.636f,
                    0.727f, 0.818f, 0.909f, 1.0f
            };

            // Run the method
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imageArray, width, height, depth, stabiliseNoiseVariance, normalizeOutput);

            // Assert that the image was returned correctly
            assertNotNull(inputImage, "Image should be retrieved successfully.");
            assertArrayEquals(expectedArray, inputImage.getImageArray(), 0.001f);
        }
    }


    @Nested
    class GetMeanAndVarBlock3DTest {

        @Test
        void testGetMeanAndVarBlock3D() {
            // Setup: create a 3D image with known values (4x4x4)
            int imageWidth = 4;
            int imageHeight = 4;
            int depth = 4;
            float[] pixels = new float[] {
                    // Slice 1 (z=0)
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f,
                    // Slice 2 (z=1)
                    17.0f, 18.0f, 19.0f, 20.0f,
                    21.0f, 22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f, 28.0f,
                    29.0f, 30.0f, 31.0f, 32.0f,
                    // Slice 3 (z=2)
                    33.0f, 34.0f, 35.0f, 36.0f,
                    37.0f, 38.0f, 39.0f, 40.0f,
                    41.0f, 42.0f, 43.0f, 44.0f,
                    45.0f, 46.0f, 47.0f, 48.0f,
                    // Slice 4 (z=3)
                    49.0f, 50.0f, 51.0f, 52.0f,
                    53.0f, 54.0f, 55.0f, 56.0f,
                    57.0f, 58.0f, 59.0f, 60.0f,
                    61.0f, 62.0f, 63.0f, 64.0f
            };

            // Define block coordinates (xStart, yStart, zStart, xEnd, yEnd, zEnd)
            int xStart = 1, yStart = 1, zStart = 1;
            int xEnd = 3, yEnd = 3, zEnd = 3; // This block is a 2x2x2 block from slice 2-3

            // Expected mean and variance for the 2x2x2 block
            float expectedMean = (22.0f + 23.0f + 26.0f + 27.0f + 38.0f + 39.0f + 42.0f + 43.0f) / 8.0f;
            float expectedVariance = (float) (Math.pow(22.0 - expectedMean, 2) + Math.pow(23.0 - expectedMean, 2)
                    + Math.pow(26.0 - expectedMean, 2) + Math.pow(27.0 - expectedMean, 2)
                    + Math.pow(38.0 - expectedMean, 2) + Math.pow(39.0 - expectedMean, 2)
                    + Math.pow(42.0 - expectedMean, 2) + Math.pow(43.0 - expectedMean, 2)) / 8.0f;

            // Run the method
            double[] result = Utils.getMeanAndVarBlock3D(pixels, imageWidth, imageHeight, xStart, yStart, zStart, xEnd, yEnd, zEnd);

            // Assert the mean and variance are correct
            assertEquals(expectedMean, (float)result[0], 0.0001f, "Mean should match expected value.");
            assertEquals(expectedVariance, (float)result[1], 0.0001f, "Variance should match expected value.");
        }
    }


    @Nested
    class GetNoiseMeanVar3DTest {

        @Test
        void testGetMeanNoiseVar3D() {
            // Setup: define dimensions and localStds array
            int imageWidth = 5;
            int imageHeight = 5;
            int imageDepth = 5;
            int blockRadiusWidth = 1;
            int blockRadiusHeight = 1;
            int blockRadiusDepth = 1;

            // Create a known localStds array (5x5x5), representing standard deviations at each pixel
            float[] localStds = new float[] {
                    // Slice 1
                    1.0f, 1.2f, 1.1f, 1.3f, 1.0f,
                    1.1f, 1.3f, 1.2f, 1.3f, 1.0f,
                    1.2f, 1.1f, 1.0f, 1.2f, 1.1f,
                    1.3f, 1.1f, 1.3f, 1.2f, 1.0f,
                    1.0f, 1.2f, 1.1f, 1.0f, 1.3f,
                    // Slice 2
                    1.0f, 1.1f, 1.2f, 1.3f, 1.1f,
                    1.2f, 1.3f, 1.0f, 1.3f, 1.1f,
                    1.3f, 1.2f, 1.1f, 1.2f, 1.3f,
                    1.1f, 1.3f, 1.2f, 1.1f, 1.0f,
                    1.2f, 1.1f, 1.3f, 1.1f, 1.2f,
                    // Slice 3
                    1.1f, 1.2f, 1.3f, 1.0f, 1.2f,
                    1.3f, 1.2f, 1.1f, 1.0f, 1.3f,
                    1.1f, 1.3f, 1.2f, 1.0f, 1.1f,
                    1.2f, 1.0f, 1.1f, 1.2f, 1.3f,
                    1.3f, 1.2f, 1.1f, 1.3f, 1.2f,
                    // Slice 4
                    1.1f, 1.3f, 1.2f, 1.0f, 1.2f,
                    1.0f, 1.3f, 1.1f, 1.3f, 1.2f,
                    1.3f, 1.1f, 1.2f, 1.1f, 1.0f,
                    1.2f, 1.0f, 1.3f, 1.1f, 1.2f,
                    1.1f, 1.2f, 1.0f, 1.2f, 1.3f,
                    // Slice 5
                    1.3f, 1.0f, 1.1f, 1.3f, 1.2f,
                    1.0f, 1.1f, 1.2f, 1.1f, 1.3f,
                    1.3f, 1.2f, 1.0f, 1.1f, 1.2f,
                    1.0f, 1.3f, 1.1f, 1.2f, 1.1f,
                    1.2f, 1.3f, 1.0f, 1.1f, 1.0f
            };

            float expectedMeanNoiseVar = 1.355f; // Example expected result

            // Run the method
            float result = Utils.getMeanNoiseVar3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, localStds);

            // Assert the result is correct
            assertEquals(expectedMeanNoiseVar, result, 0.001f, "Mean noise variance should match expected value.");
        }
    }


    @Nested
    class GetRelevanceMask3DTest {

        @Test
        void testGetRelevanceMask3D() {
            // Setup: define dimensions and input parameters
            int imageWidth = 5;
            int imageHeight = 5;
            int imageDepth = 5;
            int blockRadiusWidth = 1;
            int blockRadiusHeight = 1;
            int blockRadiusDepth = 1;
            float relevanceConstant = 1.0f;

            // Create a known localStds array (5x5x5), representing standard deviations at each pixel
            float[] localStds = new float[] {
                    // Slice 1
                    1.0f, 1.2f, 1.1f, 1.3f, 1.0f,
                    1.1f, 1.3f, 1.2f, 1.3f, 1.0f,
                    1.2f, 1.1f, 1.0f, 1.2f, 1.1f,
                    1.3f, 1.1f, 1.3f, 1.2f, 1.0f,
                    1.0f, 1.2f, 1.1f, 1.0f, 1.3f,
                    // Slice 2
                    1.0f, 1.1f, 1.2f, 1.3f, 1.1f,
                    1.2f, 1.3f, 1.0f, 1.3f, 1.1f,
                    1.3f, 1.2f, 1.1f, 1.2f, 1.3f,
                    1.1f, 1.3f, 1.2f, 1.1f, 1.0f,
                    1.2f, 1.1f, 1.3f, 1.1f, 1.2f,
                    // Slice 3
                    1.1f, 1.2f, 1.3f, 1.0f, 1.2f,
                    1.3f, 1.2f, 1.1f, 1.0f, 1.3f,
                    1.1f, 1.3f, 1.2f, 1.0f, 1.1f,
                    1.2f, 1.0f, 1.1f, 1.2f, 1.3f,
                    1.3f, 1.2f, 1.1f, 1.3f, 1.2f,
                    // Slice 4
                    1.1f, 1.3f, 1.2f, 1.0f, 1.2f,
                    1.0f, 1.3f, 1.1f, 1.3f, 1.2f,
                    1.3f, 1.1f, 1.2f, 1.1f, 1.0f,
                    1.2f, 1.0f, 1.3f, 1.1f, 1.2f,
                    1.1f, 1.2f, 1.0f, 1.2f, 1.3f,
                    // Slice 5
                    1.3f, 1.0f, 1.1f, 1.3f, 1.2f,
                    1.0f, 1.1f, 1.2f, 1.1f, 1.3f,
                    1.3f, 1.2f, 1.0f, 1.1f, 1.2f,
                    1.0f, 1.3f, 1.1f, 1.2f, 1.1f,
                    1.2f, 1.3f, 1.0f, 1.1f, 1.0f
            };

            // Calculate expected mean noise variance
            float expectedMeanNoiseVariance = Utils.getMeanNoiseVar3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, localStds);

            // Calculate the relevance mask
            float[] relevanceMask = Utils.getRelevanceMask3D(imageWidth, imageHeight, imageDepth, blockRadiusWidth,
                    blockRadiusHeight, blockRadiusDepth, localStds, relevanceConstant);

            // Verify the mask values
            for (int z = blockRadiusDepth; z < imageDepth - blockRadiusDepth; z++) {
                for (int y = blockRadiusHeight; y < imageHeight - blockRadiusHeight; y++) {
                    for (int x = blockRadiusWidth; x < imageWidth - blockRadiusWidth; x++) {
                        int index = imageWidth * imageHeight * z + y * imageWidth + x;
                        float localVariance = localStds[index] * localStds[index];
                        if (localVariance <= expectedMeanNoiseVariance * relevanceConstant) {
                            assertEquals(0.0f, relevanceMask[index], "Relevance mask value at (" + x + ", " + y + ", " + z + ") should be 0.0");
                        } else {
                            assertEquals(1.0f, relevanceMask[index], "Relevance mask value at (" + x + ", " + y + ", " + z + ") should be 1.0");
                        }
                    }
                }
            }
        }
    }


    @Nested
    class ApplyMask3DTest {

        @Test
        void testApplyMask3D() {
            // Setup: Define dimensions and initialize a test image and mask
            int imageWidth = 3;
            int imageHeight = 3;
            int imageDepth = 3;

            // Create a sample 3D image (3x3x3)
            float[] imageArray = new float[] {
                    // Slice 1
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    // Slice 2
                    10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f,
                    16.0f, 17.0f, 18.0f,
                    // Slice 3
                    19.0f, 20.0f, 21.0f,
                    22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Create a mask (1.0f for keep, 0.0f for remove)
            float[] mask = new float[] {
                    // Slice 1
                    1.0f, 0.0f, 1.0f,
                    1.0f, 1.0f, 0.0f,
                    0.0f, 1.0f, 1.0f,
                    // Slice 2
                    0.0f, 1.0f, 0.0f,
                    1.0f, 0.0f, 1.0f,
                    1.0f, 0.0f, 0.0f,
                    // Slice 3
                    1.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f,
                    1.0f, 1.0f, 1.0f
            };

            // Expected result after applying mask
            float[] expectedImageArray = new float[] {
                    // Slice 1
                    1.0f, 0.0f, 3.0f,
                    4.0f, 5.0f, 0.0f,
                    0.0f, 8.0f, 9.0f,
                    // Slice 2
                    0.0f, 11.0f, 0.0f,
                    13.0f, 0.0f, 15.0f,
                    16.0f, 0.0f, 0.0f,
                    // Slice 3
                    19.0f, 0.0f, 21.0f,
                    0.0f, 23.0f, 0.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Apply the mask
            float[] resultImageArray = Utils.applyMask3D(imageArray, imageWidth, imageHeight, imageDepth, mask);

            // Check that the result matches the expected result
            assertArrayEquals(expectedImageArray, resultImageArray, "The masked image array does not match the expected output.");
        }
    }


    @Nested
    class NormalizeImage3DTest {


        @Test
        void testNormalizeImage3DWithoutMask() {
            // Setup: Define dimensions and initialize a test image
            int imageWidth = 3;
            int imageHeight = 3;
            int imageDepth = 3;

            // Create a sample 3D image (3x3x3)
            float[] imageArray = new float[] {
                    // Slice 1
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    // Slice 2
                    10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f,
                    16.0f, 17.0f, 18.0f,
                    // Slice 3
                    19.0f, 20.0f, 21.0f,
                    22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Define border dimensions
            int borderWidth = 0;
            int borderHeight = 0;
            int borderDepth = 0;

            // Expected result after normalization (values remapped to [0, 1])
            float[] expectedNormalizedArray = new float[] {
                    // Slice 1
                    0.0f, 0.03846154f, 0.07692308f,
                    0.115384616f, 0.15384616f, 0.1923077f,
                    0.23076923f, 0.26923078f, 0.30769232f,
                    // Slice 2
                    0.34615386f, 0.3846154f, 0.42307693f,
                    0.46153846f, 0.5f, 0.53846157f,
                    0.5769231f, 0.61538464f, 0.65384614f,
                    // Slice 3
                    0.6923077f, 0.7307692f, 0.7692308f,
                    0.8076923f, 0.84615386f, 0.88461536f,
                    0.9230769f, 0.96153843f, 1.0f
            };

            // Normalize the image without mask
            float[] normalizedImage = Utils.normalizeImage3D(imageArray, imageWidth, imageHeight, imageDepth, borderWidth, borderHeight, borderDepth, null);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedImage, Utils.EPSILON, "The normalized image array does not match the expected output.");
        }

        @Test
        void testNormalizeImage3DWithMask() {
            // Setup: Define dimensions and initialize a test image with a mask
            int imageWidth = 3;
            int imageHeight = 3;
            int imageDepth = 3;

            // Create a sample 3D image (3x3x3)
            float[] imageArray = new float[] {
                    // Slice 1
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    // Slice 2
                    10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f,
                    16.0f, 17.0f, 18.0f,
                    // Slice 3
                    19.0f, 20.0f, 21.0f,
                    22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Create a mask (1.0f for keep, 0.0f for remove)
            float[] mask = new float[] {
                    // Slice 1
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    // Slice 2
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    // Slice 3
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f
            };

            // Define border dimensions
            int borderWidth = 0;
            int borderHeight = 0;
            int borderDepth = 0;

            // Expected result after normalization (same as without mask, since all values are included)
            float[] expectedNormalizedArray = new float[] {
                    // Slice 1
                    0.0f, 0.03846154f, 0.07692308f,
                    0.115384616f, 0.15384616f, 0.1923077f,
                    0.23076923f, 0.26923078f, 0.30769232f,
                    // Slice 2
                    0.34615386f, 0.3846154f, 0.42307693f,
                    0.46153846f, 0.5f, 0.53846157f,
                    0.5769231f, 0.61538464f, 0.65384614f,
                    // Slice 3
                    0.6923077f, 0.7307692f, 0.7692308f,
                    0.8076923f, 0.84615386f, 0.88461536f,
                    0.9230769f, 0.96153843f, 1.0f
            };

            // Normalize the image with mask
            float[] normalizedImage = Utils.normalizeImage3D(imageArray, imageWidth, imageHeight, imageDepth, borderWidth, borderHeight, borderDepth, mask);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedImage, Utils.EPSILON, "The normalized image array with mask does not match the expected output.");
        }

        @Test
        void testNormalizeImage3DWithMaskAndBorders() {
            // Setup: Define dimensions and initialize a test image with a mask
            int imageWidth = 3;
            int imageHeight = 3;
            int imageDepth = 3;

            // Create a sample 3D image (3x3x3)
            float[] imageArray = new float[] {
                    // Slice 1
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    // Slice 2
                    10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f,
                    16.0f, 17.0f, 18.0f,
                    // Slice 3
                    19.0f, 20.0f, 21.0f,
                    22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Create a mask (1.0f for keep, 0.0f for remove)
            float[] mask = new float[] {
                    // Slice 1
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    // Slice 2
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    // Slice 3
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f
            };

            // Define border dimensions
            int borderWidth = 1;
            int borderHeight = 1;
            int borderDepth = 1;

            // Expected result after normalization (with borders ignored)
            float[] expectedNormalizedArray = new float[] {
                    // Slice 1
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    // Slice 2
                    10.0f, 11.0f, 12.0f,
                    13.0f, 0.0f, 15.0f,
                    16.0f, 17.0f, 18.0f,
                    // Slice 3
                    19.0f, 20.0f, 21.0f,
                    22.0f, 23.0f, 24.0f,
                    25.0f, 26.0f, 27.0f
            };

            // Normalize the image with mask and borders
            float[] normalizedImage = Utils.normalizeImage3D(imageArray, imageWidth, imageHeight, imageDepth, borderWidth, borderHeight, borderDepth, mask);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedImage, Utils.EPSILON, "The normalized image array with mask and borders does not match the expected output.");
        }
    }


    // -------------------------- //
    // ---- UNSORTED METHODS ---- //
    // -------------------------- //

    @Nested
    class NormalizeArrayTest {

        @Test
        void testNormalizeArray() {
            // Setup: Create a sample array
            float[] inputArray = new float[] {3.0f, 5.0f, 1.0f, 8.0f, 2.0f};

            // Expected normalized output based on the min (1.0f) and max (8.0f)
            float[] expectedNormalizedArray = new float[] {
                    (3.0f - 1.0f) / (8.0f - 1.0f + Utils.EPSILON), // ≈ 0.25
                    (5.0f - 1.0f) / (8.0f - 1.0f + Utils.EPSILON), // ≈ 0.50
                    (1.0f - 1.0f) / (8.0f - 1.0f + Utils.EPSILON), // 0.0
                    (8.0f - 1.0f) / (8.0f - 1.0f + Utils.EPSILON), // 1.0
                    (2.0f - 1.0f) / (8.0f - 1.0f + Utils.EPSILON)  // ≈ 0.125
            };

            // Normalize the array
            float[] normalizedArray = Utils.normalizeArray(inputArray);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedArray, Utils.EPSILON, "The normalized array does not match the expected output.");
        }

        @Test
        void testNormalizeArraySingleValue() {
            // Test with an array of a single value
            float[] inputArray = new float[] {5.0f};

            // Normalizing a single value should return an array with the same value
            float[] expectedNormalizedArray = new float[] {0.0f}; // Edge case

            // Normalize the array
            float[] normalizedArray = Utils.normalizeArray(inputArray);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedArray, Utils.EPSILON, "The normalized array with a single value does not match the expected output.");
        }

        @Test
        void testNormalizeArrayWithNegativeValues() {
            // Setup: Create a sample array with negative values
            float[] inputArray = new float[] {-3.0f, -5.0f, -1.0f, -8.0f, -2.0f};

            // Find min and max
            float arrMin = Float.MAX_VALUE;
            float arrMax = -Float.MAX_VALUE;
            for(int i=0; i< inputArray.length; i++){
                arrMin = min(arrMin, inputArray[i]);
                arrMax = max(arrMax, inputArray[i]);
            }

            // Calculate expected normalized array
            float[] expectedNormalizedArray = {
                    (inputArray[0]-arrMin)/(arrMax-arrMin+Utils.EPSILON),
                    (inputArray[1]-arrMin)/(arrMax-arrMin+Utils.EPSILON),
                    (inputArray[2]-arrMin)/(arrMax-arrMin+Utils.EPSILON),
                    (inputArray[3]-arrMin)/(arrMax-arrMin+Utils.EPSILON),
                    (inputArray[4]-arrMin)/(arrMax-arrMin+Utils.EPSILON),
            };

            // Normalize the array
            float[] normalizedArray = Utils.normalizeArray(inputArray);

            // Check that the result matches the expected result
            assertArrayEquals(expectedNormalizedArray, normalizedArray, Utils.EPSILON, "The normalized array with negative values does not match the expected output.");
        }
    }


    @Nested
    class GetImageIDByTitleTest {

        @Test
        void testGetImageIDByTitle() {
            String[] titles = {"Image1", "Image2", "Image3"};
            int[] ids = {101, 102, 103};

            // Test valid title
            assertEquals(101, Utils.getImageIDByTitle(titles, ids, "Image1"));
            assertEquals(102, Utils.getImageIDByTitle(titles, ids, "Image2"));
            assertEquals(103, Utils.getImageIDByTitle(titles, ids, "Image3"));
        }

        @Test
        void testGetImageIDByTitleNotFound() {
            String[] titles = {"Image1", "Image2", "Image3"};
            int[] ids = {101, 102, 103};

            // Test title not found
            Exception exception = assertThrows(IllegalArgumentException.class, () -> {
                Utils.getImageIDByTitle(titles, ids, "NonExistentImage");
            });

            String expectedMessage = "Title not found: NonExistentImage";
            String actualMessage = exception.getMessage();
            assertTrue(actualMessage.contains(expectedMessage));
        }
    }


    @Nested
    class DisplayResults2DTest {

        @Test
        void testDisplayResults2D() {
            // Prepare a mock InputImage2D
            int width = 100;
            int height = 100;
            int size = width*height;
            float[] repetitionMap = new float[size];

            // Fill repetition map with sample values
            for (int i = 0; i < repetitionMap.length; i++) {
                repetitionMap[i] = (float) i; // Sample values for the test
            }
            Utils.InputImage2D inputImage = new Utils.InputImage2D(repetitionMap, width, height, size);

            // Call the method to display results
            Utils.displayResults2D(inputImage, repetitionMap);

            // Check if the image is displayed
            ImagePlus displayedImage = WindowManager.getCurrentImage();
            assertNotNull(displayedImage, "Image should be displayed.");

            // Verify the image properties
            assertEquals(width, displayedImage.getWidth(), "Width should match the input image width.");
            assertEquals(height, displayedImage.getHeight(), "Height should match the input image height.");

            // Check that the FloatProcessor is set up correctly
            FloatProcessor fp = (FloatProcessor) displayedImage.getProcessor();
            assertEquals(repetitionMap.length, fp.getPixelCount(), "Pixel count should match the repetition map size.");
        }
    }


    @Nested
    class DisplayResults3DTest {

        @Test
        public void testDisplayResults3D() {
            // Prepare test data
            int width = 5;
            int height = 5;
            int depth = 3;
            float[] repetitionMap = new float[width * height * depth];
            for (int z = 0; z < depth; z++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        repetitionMap[width * height * z + y * width + x] = (float) (Math.random() * 100);
                    }
                }
            }

            // Create calibration (as needed by InputImage3D)
            Calibration calibration = new Calibration();
            // Set any necessary calibration properties here if needed

            // Create InputImage3D with all required parameters
            Utils.InputImage3D inputImage = new Utils.InputImage3D(repetitionMap, width, height, depth, repetitionMap.length, calibration, 0.0f, 0.0f, 0.0f);

            // Call the method under test
            Utils.displayResults3D(inputImage, repetitionMap);

            // Check if the ImagePlus was created and is not null
            ImagePlus imp = WindowManager.getCurrentImage();
            assertNotNull(imp, "The ImagePlus should be created and not null.");

            // Check that the image stack has the correct dimensions
            assertEquals(depth, imp.getStackSize(), "The image stack should have the correct depth.");
            assertEquals(width, imp.getWidth(), "The image width should match the input width.");
            assertEquals(height, imp.getHeight(), "The image height should match the input height.");

            // Check if the LUT was applied correctly (this is more complex to test, you might need to validate visually or check properties)
            // Additional assertions regarding the LUT can be added based on your requirements

            // Clean up - close the image if you want to avoid clutter in your test suite
            imp.close();
        }
    }
}
