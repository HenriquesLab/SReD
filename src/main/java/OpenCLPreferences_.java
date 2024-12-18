/**
 *
 * Allows the user to define which device to use for OpenCL.
 *
 * @author Afonso Mendes
 *
 **/


import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLException;
import com.jogamp.opencl.CLPlatform;
import ij.IJ;
import ij.Prefs;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;

public class OpenCLPreferences_ implements PlugIn {

    static private CLPlatform clPlatformMaxFlop;

    @Override
    public void run(String s) {


        // ------------------------------ //
        // ---- Check OpenCL devices ---- //
        // ------------------------------ //

        // Initialize OpenCL
        CLPlatform[] allPlatforms = CLPlatform.listCLPlatforms();

        try {
            allPlatforms = CLPlatform.listCLPlatforms();
        } catch (CLException ex) {
            IJ.log("Something went wrong while initialising OpenCL.");
            throw new RuntimeException("Something went wrong while initialising OpenCL.");
        }


        // ----------------------------------- //
        // ---- Make list of device names ---- //
        // ----------------------------------- //

        CLDevice[] allCLdeviceOnThisPlatform = new CLDevice[0];

        for (CLPlatform allPlatform : allPlatforms) {
            allCLdeviceOnThisPlatform = allPlatform.listCLDevices();
        }

        String[] deviceNames = new String[allCLdeviceOnThisPlatform.length];
        for (int i=0; i<allCLdeviceOnThisPlatform.length; i++) {
            deviceNames[i] = allCLdeviceOnThisPlatform[i].getName();
            System.out.println(deviceNames[i]);
        }


        // -------------------- //
        // ---- Dialog box ---- //
        // -------------------- //

        String[] sizes = {"64", "128", "256", "512", "1024" , "2048"}; // Define compute block sizes
        String[] gatOptions = {"Simplex", "Quadtree/Octree"}; // Define methods to estimate GAT parameters

        GenericDialog gd = new GenericDialog("OpenCL Preferences");
        gd.addChoice("Preferred device", deviceNames, deviceNames[0]);
        gd.addChoice("Max. compute block size", sizes, sizes[0]);
        gd.addChoice("GAT parameters estimation", gatOptions, gatOptions[1]);
        gd.showDialog();


        // ------------------------ //
        // ---- Set preference ---- //
        // ------------------------ //

        // Prefered OpenCL device
        String prefDevice = gd.getNextChoice();
        for (int i=0; i<deviceNames.length; i++) {
            if (deviceNames[i].equals(prefDevice)) {
                Prefs.set("SReD.OpenCL.device", deviceNames[i]);
            }
        }

        // Prefered compute block size
        String prefBlockSize = gd.getNextChoice();
        for (int i=0; i<sizes.length; i++) {
            if (sizes[i].equals(prefBlockSize)) {
                int number = Integer.parseInt(prefBlockSize);
                Prefs.set("SReD.OpenCL.blockSize", number);
            }
        }

        // Prefered method to estimate GAT parameters
        String prefGATMethod = gd.getNextChoice();
        for(int i=0; i<gatOptions.length; i++){
            if(gatOptions[i].equals(prefGATMethod)){
                Prefs.set("SReD.OpenCL.gatMethod", gatOptions[i]);
            }
        }

    }
}
