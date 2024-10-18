/**
 * This class calculates global repetition maps for 2D data.
 *
 * @author Afonso Mendes
 */
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import ij.IJ;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;

import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;

public class GlobalRepetition2D_ implements PlugIn {

    // Define constants for metric choices
    public static final String[] METRICS = {
            "Pearson's R",
            "Cosine similarity",
            "SSIM",
            "NRMSE",
            "Abs. diff. of StdDevs"
    };

    @Override
    public void run(String arg) {


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
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Global Repetition (2D)");
        gd.addChoice("Image:", titles, titles[0]);
        gd.addNumericField("Block width (px):", 5);
        gd.addNumericField("Block height (px):", 5);
        gd.addCheckbox("Time-lapse?", false);
        gd.addNumericField("Relevance constant:", 0.0f);
        gd.addChoice("Metric:", METRICS, METRICS[0]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);
        gd.addHelp("https://github.com/HenriquesLab/SReD/wiki");
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get dialog parameters
        String imageTitle = gd.getNextChoice();
        int imgID = Utils.getImageIDByTitle(titles, ids, imageTitle);
        int blockWidth = (int) gd.getNextNumber();
        int blockHeight = (int) gd.getNextNumber();
        boolean isTimelapse = gd.getNextBoolean();
        float relevanceConstant = (float) gd.getNextNumber();
        String metric = gd.getNextChoice();
        boolean normalizeOutput = gd.getNextBoolean();
        boolean useDevice = gd.getNextBoolean();

        // Check if block dimensions are odd, otherwise kill program
        if (blockWidth % 2 == 0 || blockHeight % 2 == 0) {
            IJ.error("Patch dimensions must be odd. Please try again.");
            return;
        }

        IJ.log("--------");
        IJ.log("SReD is running, please wait...");

        // Start timer
        long start = System.currentTimeMillis();

        // Calculate block radius
        int blockRadiusWidth = blockWidth /2; // Patch radius (x-axis)
        int blockRadiusHeight = blockHeight /2; // Patch radius (y-axis)

        // Get final block size (after removing pixels outside inbound circle/ellipse)
        int blockSize = 0;
        for(int y=0; y<blockHeight; y++){
            for (int x=0; x<blockWidth; x++) {
                float dx = (float)(x-blockRadiusWidth);
                float dy = (float)(y-blockRadiusHeight);
                if(((dx*dx)/(float)(blockRadiusWidth*blockRadiusWidth))+((dy*dy)/(float)(blockRadiusHeight*blockRadiusHeight)) <= 1.0f){
                    blockSize++;
                }
            }
        }

        // Get reference image and some parameters
        Utils.InputImage2D inputImage = Utils.getInputImage2D(imgID, true, true);

        // Initialize OpenCL, and retrieve OpenCL context, device and queue
        CLUtils.OpenCLResources openCLResources = CLUtils.getOpenCLResources(useDevice);
        CLContext context = openCLResources.getContext();
        CLDevice device = openCLResources.getDevice();
        CLCommandQueue queue = openCLResources.getQueue();

        // Calculate local statistics
        CLUtils.CLLocalStatistics localStatistics = CLUtils.getLocalStatistics2D(context, device, queue, inputImage,
                blockWidth, blockHeight, Utils.EPSILON);

        // Calculate relevance mask
        Utils.RelevanceMask relevanceMask = Utils.getRelevanceMask(inputImage.getImageArray(),
                inputImage.getWidth(), inputImage.getHeight(), blockRadiusWidth, blockRadiusHeight,
                localStatistics.getLocalStds(), relevanceConstant);

        // Calculate the number of structurally relevant pixels
        float nPixels = 0.0f;
        for (int y=blockRadiusHeight; y<inputImage.getHeight()-blockRadiusHeight; y++) {
            for(int x=blockRadiusWidth; x<inputImage.getWidth()-blockRadiusWidth; x++) {
                if (relevanceMask.getRelevanceMask()[y*inputImage.getWidth()+x] > 0.0f) {
                    nPixels += 1.0f;
                }
            }
        }

        // Calculate global repetition map
        float[] repetitionMap;
        repetitionMap = CLUtils.calculateGlobalRepetitionMap2D(metric, inputImage, blockWidth, blockHeight, blockSize,
                localStatistics, relevanceMask, nPixels, normalizeOutput, openCLResources);

        // Display results
        Utils.displayResults2D(inputImage, repetitionMap);

        // Stop timer
        long elapsedTime = System.currentTimeMillis() - start;
        IJ.log("Finished!");
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");

    }
}
