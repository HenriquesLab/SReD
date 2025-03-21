/**
 *
 * Returns a 3D repetition map where each pixel value represents the repetition score between the block centered around that pixel and a reference block.
 *
 * @author Afonso Mendes
 * @version 2023.11.01
 *
 */

import ij.*;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;

public class BlockRepetition3D_ implements PlugIn {

    // Define constants for metric choices
    public static final String[] METRICS = {
            "Pearson's R",
            "SSIM",
            "NRMSE (inverted)",
            "Abs. Diff. of StdDevs."
    };

    // Define GAT parameter estimation methods
    public static final String[] GATMETHODS = {
            "Simplex",
            "Quad/Octree"
    };

    @Override
    public void run(String s) {

        // Install the SReD custom LUT
        installLUT.run();


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        // Get all open image titles
        int nImages = getImageCount();
        if (nImages < 2) {
            IJ.error("Not enough images found. You need at least two.");
            throw new IllegalArgumentException("Not enough images found. You need at least two.");
        }

        // Get image IDs from titles
        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for (int i = 0; i < nImages; i++) {
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Block Repetition (3D)");
        gd.addMessage("Find repetitions of a single reference block in 3D data.\n");
        gd.addChoice("Block:", titles, titles[1]);
        gd.addChoice("Image:", titles, titles[0]);
        gd.addNumericField("Relevance constant:", 0.0f, 3);
        gd.addChoice("Metric:", METRICS, METRICS[0]);
        gd.addCheckbox("Stabilise noise variance?", true);
        gd.addChoice("GAT parameter estimation:", GATMETHODS, GATMETHODS[0]);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use device from preferences?", false);
        gd.addHelp("https://github.com/HenriquesLab/SReD/wiki");
        gd.showDialog();
        if (gd.wasCanceled()) return;

        // Get parameter values from dialog box
        String blockTitle = gd.getNextChoice();
        int blockID = Utils.getImageIDByTitle(titles, ids, blockTitle);
        String imgTitle = gd.getNextChoice();
        int imgID = Utils.getImageIDByTitle(titles, ids, imgTitle);
        float relevanceConstant = (float) gd.getNextNumber();
        String metric = gd.getNextChoice();
        boolean stabilizeNoiseVariance = gd.getNextBoolean();
        String gatMethod = gd.getNextChoice();
        boolean normalizeOutput = gd.getNextBoolean();
        boolean useDevice = gd.getNextBoolean();

        IJ.log("--------");
        IJ.log("SReD is running, please wait...");

        // Start timer
        long start = System.currentTimeMillis();

        // Get reference block and some parameters
        Utils.ReferenceBlock3D referenceBlock = Utils.getReferenceBlock3D(blockID);

        // Get variance-stabilised and normalised input image
        Utils.InputImage3D inputImage = Utils.getInputImage3D(imgID, stabilizeNoiseVariance, gatMethod, true);

        // Check if block dimensions are not larger than the image, otherwise kill program
        if (referenceBlock.getWidth() > inputImage.getWidth() || referenceBlock.getHeight() > inputImage.getHeight() || referenceBlock.getDepth() > inputImage.getDepth()) {
            IJ.error("Block dimensions can't be larger than image dimensions.");
            throw new IllegalArgumentException("Block dimensions must be smaller than image dimensions.");
        }

        // Calculate repetition map
        float[] repetitionMap = CLUtils.calculateBlockRepetitionMap3D(metric, inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);

        // Display results
        Utils.displayResults3D(inputImage, repetitionMap);

        // Stop timer
        long elapsedTime = System.currentTimeMillis() - start;

        IJ.log("Finished!");
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");

    }
}