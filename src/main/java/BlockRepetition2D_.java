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
        gd.addCheckbox("Stabilise noise variance?", true);
        gd.addChoice("GAT parameter estimation:", GATMETHODS, GATMETHODS[0]);
        gd.addCheckbox("Timelapse?", false);
        gd.addCheckbox("Normalize output?", true);
        gd.addCheckbox("Use OpenCL device from preferences?", false);
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
        boolean stabiliseNoiseVariance = gd.getNextBoolean();
        String gatMethod = gd.getNextChoice();
        boolean isTimelapse = gd.getNextBoolean();
        boolean normalizeOutput = gd.getNextBoolean();
        boolean useDevice = gd.getNextBoolean();

        IJ.log("--------");
        IJ.log("SReD is running, please wait...");

        // Start timer
        long start = System.currentTimeMillis();

        // Get reference block and some parameters
        Utils.ReferenceBlock2D referenceBlock = Utils.getReferenceBlock2D(blockID);

        // Process
        if(isTimelapse){
            // Get variance-stabilised and normalised input image stack
            Utils.InputImage3D inputImage = Utils.getInputImage3D(imgID, true, true);
            float gain = inputImage.getGain();
            float sigma = inputImage.getSigma();
            float offset = inputImage.getOffset();
            float[] repetitionMap = new float[inputImage.getSize()];

            // Calculate block repetition map
            for(int t=0; t<inputImage.getDepth(); t++){ // For each time frame
                IJ.log("Processing frame " + t);

                // Make temporary copy of time frame
                float[] imageArray = new float[inputImage.getWidth()* inputImage.getHeight()];
                for(int y=0; y<inputImage.getHeight(); y++){
                    for(int x=0; x<inputImage.getWidth(); x++){
                        imageArray[y*inputImage.getWidth()+x] = inputImage.getImageArray()[inputImage.getWidth()*inputImage.getHeight()*t+y*inputImage.getWidth()+x];
                    }
                }

                // Get InputImage2D object containing the time frame (already stabilised from the origin)
                Utils.InputImage2D tempImage = Utils.getInputImage2D(imageArray, inputImage.getWidth(),
                        inputImage.getHeight(), false, 0.0f, 0.0f, 0.0f, true);

                // Calculate block repetition map of the time frame
                float[] tempRepetitionMap;
                tempRepetitionMap = CLUtils.calculateBlockRepetitionMap2D(metric, tempImage, referenceBlock,
                        relevanceConstant, normalizeOutput, useDevice);

                // Save result to 3D repetition map
                for(int y=0; y<inputImage.getHeight(); y++){
                    for(int x=0; x<inputImage.getWidth(); x++){
                        repetitionMap[inputImage.getWidth()*inputImage.getHeight()*t+y*inputImage.getWidth()+x] = tempRepetitionMap[y*inputImage.getWidth()+x];
                    }
                }
            }

            // Display results
            Utils.InputImage3D finalRepetitionMap = Utils.getInputImage3D(repetitionMap, inputImage.getWidth(),
                    inputImage.getHeight(), inputImage.getDepth(), false, normalizeOutput);

            Utils.displayResults3D(finalRepetitionMap, repetitionMap);

        }else {
            // Get variance-stabilised and normalised input image
            Utils.InputImage2D inputImage = Utils.getInputImage2D(imgID, stabiliseNoiseVariance, gatMethod, true);

            // Calculate block repetition map
            float[] repetitionMap;
            repetitionMap = CLUtils.calculateBlockRepetitionMap2D(metric, inputImage, referenceBlock, relevanceConstant, normalizeOutput, useDevice);

            // Display results
            Utils.displayResults2D(inputImage, repetitionMap);
        }
        // Stop timer
        long elapsedTime = System.currentTimeMillis() - start;

        IJ.log("Finished!");
        IJ.log("Elapsed time: " + elapsedTime/1000 + " sec");
        IJ.log("--------");
    }
}

