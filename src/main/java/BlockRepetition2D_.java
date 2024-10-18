/**
 * This class calculates block repetition maps for 2D data.
 * It provides a user interface to select the reference block, input image,
 * relevance constant, and metrics for calculating the repetition map.
 *
 * @author Afonso Mendes
 */
import ij.IJ;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;
import static ij.WindowManager.getIDList;
import static ij.WindowManager.getImageCount;

public class BlockRepetition2D_ implements PlugIn {

    // Define constants for metric choices
    public static final String[] METRICS = {
            "Pearson's R",
            "Cosine similarity",
            "SSIM",
            "NRMSE (inverted)",
            "Abs. Diff. of StdDevs."
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
            return;
        }

        // Get image IDs from titles
        int[] ids = getIDList();
        String[] titles = new String[nImages];
        for (int i = 0; i < nImages; i++) {
            titles[i] = WindowManager.getImage(ids[i]).getTitle();
        }

        // Initialize dialog box
        NonBlockingGenericDialog gd = new NonBlockingGenericDialog("SReD: Block Repetition (2D)");
        gd.addMessage("Find repetitions of a single reference block in 2D data.\n");
        gd.addChoice("Block:", titles, titles[1]);
        gd.addChoice("Image:", titles, titles[0]);
        gd.addNumericField("Relevance constant:", 0.0f, 3);
        gd.addChoice("Metric:", METRICS, METRICS[0]);
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
        boolean normalizeOutput = gd.getNextBoolean();
        boolean useDevice = gd.getNextBoolean();

        IJ.log("--------");
        IJ.log("SReD is running, please wait...");

        // Start timer
        long start = System.currentTimeMillis();

        // Get reference block and some parameters
        Utils.ReferenceBlock2D referenceBlock = Utils.getReferenceBlock2D(blockID);

        // Get variance-stabilised and normalised input image
        Utils.InputImage2D inputImage = Utils.getInputImage2D(imgID, true, true);

        // Calculate block repetition map
        float[] repetitionMap;
        repetitionMap = CLUtils.calculateBlockRepetitionMap2D(metric, inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);

        // Display results
        Utils.displayResults2D(inputImage, repetitionMap);

        // Stop timer
        long elapsedTime = System.currentTimeMillis() - start;

        IJ.log("Finished!");
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }
}

