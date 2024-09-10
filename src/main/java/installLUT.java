import ij.*;
import java.io.*;
import java.nio.file.*;

public class installLUT {

    public static void run() {
        // Path to the LUTs directory in Fiji
        String fijiLutsDir = IJ.getDirectory("imagej") + "luts/";

        // Name of the custom LUT
        String lutName = "sred-jet.lut";

        // Full path to the LUT in Fiji's luts directory
        File lutFile = new File(fijiLutsDir, lutName);

        // Check if the LUT already exists
        if (!lutFile.exists()) {
            try {
                // Load the LUT from the plugin resources
                InputStream lutStream = installLUT.class.getResourceAsStream("/luts/" + lutName);

                if (lutStream == null) {
                    //IJ.error("LUT file not found in plugin resources!");
                    return;
                }

                // Create the LUT file in Fiji's luts directory
                Files.copy(lutStream, lutFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                //IJ.showMessage("LUT installed successfully to " + fijiLutsDir);

            } catch (IOException e) {
                //IJ.error("Failed to install LUT: " + e.getMessage());
            }
        } else {
            //IJ.showMessage("LUT already installed: " + lutFile.getAbsolutePath());
        }
    }
}

