/**
 * This class calculates block repetition maps for 3D data.
 *
 * @author Afonso Mendes
 */
import com.jogamp.opencl.*;
import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;

public class GlobalRepetition3D_ implements PlugIn {

    // Define constants for metric choices
    public static final String[] METRICS = {
            "Pearson's R",
            "Cosine similarity",
            "SSIM",
            "NRMSE",
            "Abs. diff. of StdDevs"
    };

    // Define GAT parameter estimation methods
    public static final String[] GATMETHODS = {
            "Simplex",
            "Quad/Octree"
    };

    @Override
    public void run(String s) {


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Get all open image titles
        int nImages = getImageCount();
        if (nImages < 1) {
            IJ.error("No images found. Open an image and try again.");
            return;
        }

        // Get image IDs from titles
        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for (int i = 0; i < nImages; i++) {
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Global Repetition (3D)");
        gd.addChoice("Image:", titles, titles[0]);
        gd.addNumericField("Block width (px):", 5);
        gd.addNumericField("Block height (px):", 5);
        gd.addNumericField("Block depth (px):", 5);
        gd.addNumericField("Relevance constant:", 0.0f);
        gd.addChoice("Metric:", METRICS, METRICS[0]);
        gd.addCheckbox("Stabilise noise variance?", true);
        gd.addChoice("GAT parameter estimation:", GATMETHODS, GATMETHODS[1]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);
        gd.addHelp("https://github.com/HenriquesLab/SReD/wiki");
        gd.showDialog();
        if(gd.wasCanceled()) return;

        // Get dialog parameters
        String imageTitle = gd.getNextChoice();
        int imageID = Utils.getImageIDByTitle(titles, ids, imageTitle);
        int blockWidth = (int) gd.getNextNumber();
        int blockHeight = (int) gd.getNextNumber();
        int blockDepth = (int) gd.getNextNumber();
        float relevanceConstant = (float) gd.getNextNumber();
        String metric = gd.getNextChoice();
        boolean stabiliseNoiseVariance = gd.getNextBoolean();
        String gatMethod = gd.getNextChoice();
        boolean normalizeOutput = gd.getNextBoolean();
        boolean useDevice = gd.getNextBoolean();

        // Check if block dimensions are odd, otherwise kill program
        if(blockWidth%2==0 || blockHeight%2==0 || blockDepth%2==0) {
            IJ.error("Block dimensions must be odd. Please try again.");
            return;
        }

        // Check if block has at least 3 slices, otherwise kill program
        if(blockDepth<3) {
            IJ.error("Block depth must be at least 3. Please try again.");
            return;
        }

        IJ.log("--------");
        IJ.log("SReD is running, please wait...");

        // Start timer
        long start = System.currentTimeMillis();

        // Calculate block radius
        int blockRadiusWidth = blockWidth/2; // Patch radius (x-axis)
        int blockRadiusHeight = blockHeight/2; // Patch radius (y-axis)
        int blockRadiusDepth = blockDepth/2; // Patch radius (z-axis)

        // Get final block size (after removing pixels outside inbound sphere/spheroid)
        int blockSize = 0;
        for(int z=0; z<blockDepth; z++) {
            for (int y=0; y<blockHeight; y++) {
                for (int x=0; x<blockWidth; x++) {
                    float dx = (float)(x-blockRadiusWidth);
                    float dy = (float)(y-blockRadiusHeight);
                    float dz = (float)(z-blockRadiusDepth);
                    if (((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight))+((dz*dz)/(float)(blockRadiusDepth*blockRadiusDepth)) <= 1.0f) {
                        blockSize++;
                    }
                }
            }
        }

        // Get reference image and some parameters
        Utils.InputImage3D inputImage = Utils.getInputImage3D(imageID, stabiliseNoiseVariance, gatMethod, true);

        // Initialize OpenCL
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLUtils.CLLocalStatistics localStatistics = CLUtils.getLocalStatistics3D(openCLResources, inputImage,
                blockRadiusWidth, blockRadiusHeight, blockRadiusDepth, blockSize,
                Utils.EPSILON);

        // Calculate relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage.getImageArray(),
                inputImage.getWidth(), inputImage.getHeight(), blockRadiusWidth, blockRadiusHeight,
                localStatistics.getLocalStds(), relevanceConstant);

        // Calculate the number of structurally relevant pixels
        float nPixels = 0.0f;
        for(int z=blockRadiusDepth; z<inputImage.getDepth()-blockRadiusDepth; z++) {
            for (int y=blockRadiusHeight; y<inputImage.getHeight()-blockRadiusHeight; y++) {
                for (int x=blockRadiusWidth; x<inputImage.getWidth()-blockRadiusWidth; x++) {
                    int index = inputImage.getWidth()*inputImage.getHeight()*z+y* inputImage.getWidth()+x;
                    if (relevanceMask.getRelevanceMask()[index] > 0.0f) {
                        nPixels += 1.0f;
                    }
                }
            }
        }

        // Calculate global repetition map
        float[] repetitionMap;
        repetitionMap = CLUtils.calculateGlobalRepetitionMap3D(metric, inputImage, blockWidth, blockHeight, blockDepth,
                blockSize, localStatistics, relevanceMask, nPixels, normalizeOutput, openCLResources);

        // Display results
        Utils.displayResults3D(inputImage, repetitionMap);

        // Stop timer
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Finished!");
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");

    }
}








