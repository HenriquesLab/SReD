import com.jogamp.opencl.*;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.NonBlockingGenericDialog;
import ij.plugin.PlugIn;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;

public class GPUAnalysisExample_ implements PlugIn {

    // OpenCL formats
    static private CLContext context; // to setup the OpenCL environment
    static private CLCommandQueue queue; // a queue for the sent GPU commands
    static private CLProgram programExample; // to setup the programme in the GPU
    static private CLKernel kernelExample; // to setup a kernel

    private CLBuffer<IntBuffer> clBufferPixelsIn;
    private CLBuffer<FloatBuffer> clBufferPixelsOut;
    private int width, height;

    // --- Constructor ---
    public void startCL() {
        IJ.log("--------");
        context = CLContext.create();
        System.out.println("created " + context);
    }

    // -- Check devices --
    public CLDevice[] checkDevices() {

        CLDevice[] allCLdevice = context.getDevices();

        for (int i = 0; i < allCLdevice.length; i++) {
            IJ.log("Device #" + i);
            IJ.log("Max clock: " + allCLdevice[i].getMaxClockFrequency() + " MHz");
            IJ.log("Max cores: " + allCLdevice[i].getMaxComputeUnits() + " cores");
            IJ.log("Device type: " + allCLdevice[i].getType());
            IJ.log("Device name: " + allCLdevice[i].getName());
            IJ.log("--------");
        }
        return allCLdevice;
    }

    // --- Initialization method ---
    public void initialise(int width, int height, CLDevice chosenDevice) {

        this.width = width;
        this.height = height;

        if (chosenDevice == null) {
            //IJ.log("Looking for the fastest device...");
            System.out.println("Looking for the fastest device...");
            chosenDevice = context.getMaxFlopsDevice();
        }

        //System.out.println("using " + chosenDevice);
        IJ.log("Using " + chosenDevice.getName());

        // fill the buffers
        clBufferPixelsIn = context.createIntBuffer(width * height, READ_ONLY);
        clBufferPixelsOut = context.createFloatBuffer(width * height, WRITE_ONLY); // potentially READ_WRITE

        String programString = getResourceAsString(GPUAnalysisExample_.class, "test.cl");

    }

    @Override
    public void run(String s) {

        // Grab image
        ImagePlus imp = WindowManager.getCurrentImage();

        int w = imp.getWidth();
        int h = imp.getHeight();
        int wh = w*h;

        initialise(w, h, null);
        ImageStack ims = imp.getImageStack();

        IJ.log("Uploading raw data to GPU...");
        fillBuffer(clBufferPixelsIn, ims);
        queue.putWriteBuffer(clBufferPixelsIn, false);




    }

    // Reads a kernel from the resources
    public static String getResourceAsString(Class c, String resourceName) {
        InputStream programStream = c.getResourceAsStream("/" + resourceName);
        String programString = "";

        try {
            programString = inputStreamToString(programStream);
        } catch (IOException var5) {
            var5.printStackTrace();
        }

        return programString;
    }

    private static String inputStreamToString(InputStream inputStream) throws IOException {
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];

        int length;
        while((length = inputStream.read(buffer)) != -1) {
            result.write(buffer, 0, length);
        }

        return result.toString("UTF-8");
    }

    public static void fillBuffer(CLBuffer<IntBuffer> clBuffer, ImageStack ims) {
        IntBuffer buffer = (IntBuffer) clBuffer.getBuffer();
        int nSlices = ims.getSize();

        for(int s = 1; s <= nSlices; ++s) {
            int[] data = (int[])(ims.getProcessor(s).convertToShortProcessor().getPixels());
            int fOffset = (s - 1) * data.length;

            for(int n = 0; n < data.length; ++n) {
                buffer.put(n + fOffset, data[n]);
            }
        }

    }

    public static void grabBuffer(CLBuffer<FloatBuffer> clBuffer, float[] data, boolean NaN2Zero) {
        FloatBuffer buffer = (FloatBuffer)clBuffer.getBuffer();

        for(int n = 0; n < data.length; ++n) {
            data[n] = buffer.get(n);
            if (NaN2Zero && Float.isNaN(data[n])) {
                data[n] = 0.0F;
            }
        }

    }
}
