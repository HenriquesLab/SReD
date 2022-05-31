#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define std_x $STD_X$

float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchPhaseCorrelation(
    global double* ref_dft_real,
    global double* ref_dft_imag,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* phase_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    double PI = 3.14159265358979323846f;
    double EPSILON = 0.00001f;

    // For each comparison pixel...
    double weight = 0.0f;

    // Get comparison patch Y
    double comp_patch[patch_size] = {0.0};
    double meanSub_y[patch_size] = {0.0};

    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = (double) ref_pixels[j*w+i];
            meanSub_y[comp_counter] = (comp_patch[comp_counter] - (double) local_means[gy*w+gx]) / ((double) local_stds[gy*w+gx] + EPSILON);
            comp_counter++;
        }
    }

    // Calculate 2D Discrete Fourier Transform
    double comp_dft_real[patch_size] = {0.0};
    double comp_dft_imag[patch_size] = {0.0};

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    comp_dft_real[j*bW+i] += (meanSub_y[jj*bW+ii] * cos(2*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                    comp_dft_imag[j*bW+i] -= (meanSub_y[jj*bW+ii] * sin(2*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                }
            }
        }
    }

    // Get comparison patch complex conjugate
    double comp_dft_conj[patch_size] = {0.0f};
    for(int i=0; i<patch_size; i++){
        comp_dft_conj[i] = (-1) * comp_dft_imag[i];
    }

    // Calculate cross-power spectrum
    double cross_spectrum_real[patch_size] = {0.0f};
    double cross_spectrum_imag[patch_size] = {0.0f};
    double multReal = 0.0f;
    double multImag = 0.0f;
    double multRealAbs = 0.0f;
    double multImagAbs = 0.0f;
    for(int i=0; i<patch_size; i++){
        multReal = ref_dft_real[i] * comp_dft_real[i] - ref_dft_imag[i] * comp_dft_conj[i];
        multImag = ref_dft_real[i] * comp_dft_conj[i] + ref_dft_imag[i] * comp_dft_real[i];

        multRealAbs = fabs((double)multReal);
        multImagAbs = fabs((double)multImag);

        cross_spectrum_real[i] = ((multReal * multRealAbs) + (multImag * multImagAbs)) / ((multRealAbs * multRealAbs) + (multImagAbs * multImagAbs) + EPSILON);
        cross_spectrum_imag[i] = ((multImag * multRealAbs) - (multReal * multImagAbs)) / ((multRealAbs * multRealAbs) + (multImagAbs * multImagAbs) + EPSILON);
    }

    // Calculate normalized cross-correlation by calculating the inverse DFT of the cross-power spectrum
    double cross_corr_real[patch_size] = {0.0f};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    cross_corr_real[j*bW+i] += (cross_spectrum_real[jj*bW+ii] * cos(2*PI*((1*i*ii/bW) + (1*j*jj/bH))) - cross_spectrum_imag[jj*bW+ii] * sin(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) / sqrt((double)patch_size);
                }
            }
        }
    }

    // Determine the maximum value of the cross-correlation and get peak coordinates
    double max_value = 0.0f;
    int x_coord = 0;
    int y_coord = 0;
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            double value = cross_corr_real[j*w+i];
            if(value > max_value){
                max_value = value;
                x_coord = i+1; // +1 to change the index range, so that zero coordinates represent no correlation
                y_coord = j+1; // +1 to change the index range, so that zero coordinates represent no correlation
            }
        }
    }
    // Calculate weight
    weight = (double) getExpDecayWeight(std_x, local_stds[gy*w+gx]);

    // Calculate Euclidean distance of peak coordinates to the origin
    phase_map[gy*w+gx] = (float) sqrt(((double) x_coord * (double) x_coord) + ((double) y_coord * (double) y_coord)); // Euclidean displacement (distance from origin)
}

// ---- USER FUNCTIONS ----
float getExpDecayWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))
    float weight = 0;

    if(ref == comp){
        weight = 1;
    }else{
        weight = 1-(fabs(ref-comp)/(ref+comp));
    }
    return weight;

}