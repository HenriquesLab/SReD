
import java.io.IOException;
import com.aparapi.natives.NativeLoader;



public class gputest {
    public static void main(String[] args) {
        try {
            NativeLoader.load();
            System.out.println("Aparapi JNI loaded successfully.");
        } catch (final IOException e) {
            System.out.println("Check your environment. Failed to load aparapi native library "
                    + " or possibly failed to locate opencl native library (opencl.dll/opencl.so)."
                    + " Ensure that OpenCL is in your PATH (windows) or in LD_LIBRARY_PATH (linux).");
        }
    }
}
