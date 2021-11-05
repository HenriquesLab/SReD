import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.aparapi.internal.kernel.KernelPreferences;
import com.aparapi.natives.NativeLoader;

import java.io.IOException;


public class DeviceInfo {
    public static void main(String[] args) {
        try {
            NativeLoader.load();
            System.out.println("Aparapi JNI loaded successfully.");
        } catch (final IOException e) {
            System.out.println("Check your environment. Failed to load aparapi native library "
                    + " or possibly failed to locate opencl native library (opencl.dll/opencl.so)."
                    + " Ensure that OpenCL is in your PATH (windows) or in LD_LIBRARY_PATH (linux).");
        }

        KernelPreferences preferences = KernelManager.instance().getDefaultPreferences();
        System.out.println("-- Devices in preferred order --");
        for (Device device : preferences.getPreferredDevices(null)) {
            System.out.println("----------");
            System.out.println(device);
        }
    }

}
