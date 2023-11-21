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

        GenericDialog gd = new GenericDialog("OpenCL Preferences");
        gd.addChoice("Preferred device", deviceNames, deviceNames[0]);
        gd.showDialog();


        // ------------------------ //
        // ---- Set preference ---- //
        // ------------------------ //

        String choice = gd.getNextChoice();

        for (int i=0; i<deviceNames.length; i++) {
            if (deviceNames[i].equals(choice)) {
                Prefs.set("SReD.OpenCL.device", deviceNames[i]);
            }
        }

    }
}
