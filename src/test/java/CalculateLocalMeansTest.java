import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.jogamp.opencl.CLBuffer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.FloatBuffer;

public class CalculateLocalMeansTest {

    private CalculateLocalMeans_ calculateLocalMeans;
    private CLBuffer<FloatBuffer> mockClRefPixels;
    private CLBuffer<FloatBuffer> mockClLocalMeans;
    private CLBuffer<FloatBuffer> mockClLocalStds;

    @BeforeEach
    public void setUp() {
        calculateLocalMeans = new CalculateLocalMeans_();
        mockClRefPixels = mock(CLBuffer.class);
        mockClLocalMeans = mock(CLBuffer.class);
        mockClLocalStds = mock(CLBuffer.class);
    }

    @Test
    public void testFillBufferWithFloatArray() {
        // Mocking FloatBuffer
        FloatBuffer mockBuffer = mock(FloatBuffer.class);
        when(mockClRefPixels.getBuffer()).thenReturn(mockBuffer);

        float[] pixels = {1.0f, 2.0f, 3.0f};

        // Calling the method
        CalculateLocalMeans_.fillBufferWithFloatArray(mockClRefPixels, pixels);

        // Verifying interactions with FloatBuffer
        verify(mockBuffer, times(1)).put(0, 1.0f);
        verify(mockBuffer, times(1)).put(1, 2.0f);
        verify(mockBuffer, times(1)).put(2, 3.0f);
    }
}
