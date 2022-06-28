#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define center_x $CENTER_X$
#define center_y $CENTER_Y$
#define bRW $BRW$
#define bRH $BRH$

float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPatchPhaseCorrelation(
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

    double PI = 3.14159265358979323846;
    double EPSILON = 0.00001;

    // Get reference patch
    double ref_patch[patch_size] = {0.0};
    double ref_mean = (double) local_means[center_y*w+center_x];
    double ref_std = (double) local_stds[center_y*w+center_x];

    int counter = 0;
    for(int j=center_y-bRH; j<=center_y+bRH; j++){
        for(int i=center_x-bRW; i<=center_x+bRW; i++){
            ref_patch[counter] = ((double) ref_pixels[j*w+i] - ref_mean);
            counter++;
        }
    }

    // Calculate reference 2D DFT
    double ref_dft_real[patch_size] = {0.0};
    double ref_dft_imag[patch_size] = {0.0};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    ref_dft_real[j*bW+i] += (ref_patch[jj*bW+ii] * cos(2.0*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                    ref_dft_imag[j*bW+i] -= (ref_patch[jj*bW+ii] * sin(2.0*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                }
            }
        }
    }

    // Get comparison patch
    double comp_patch[patch_size] = {0.0};
    double comp_mean = (double) local_means[gy*w+gx];
    double comp_std = (double) local_stds[gy*w+gx];

    counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[counter] = ((double) ref_pixels[j*w+i] - comp_mean) ;
            counter++;
        }
    }

    // Calculate 2D Discrete Fourier Transform
    double comp_dft_real[patch_size] = {0.0};
    double comp_dft_imag[patch_size] = {0.0};

    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    comp_dft_real[j*bW+i] += (comp_patch[jj*bW+ii] * cos(2*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                    comp_dft_imag[j*bW+i] -= (comp_patch[jj*bW+ii] * sin(2*PI*((i*ii/bW) + (j*jj/bH)))) / sqrt((double)patch_size);
                }
            }
        }
    }

    // Get comparison patch complex conjugate
    double comp_dft_conj[patch_size] = {0.0};
    for(int i=0; i<patch_size; i++){
        comp_dft_conj[i] = (-1.0) * comp_dft_imag[i];
    }

    // Calculate cross-power spectrum
    double cross_spectrum_real[patch_size] = {0.0};
    double cross_spectrum_imag[patch_size] = {0.0};
    double multReal = 0.0;
    double multImag = 0.0;
    double multRealAbs = 0.0;
    double multImagAbs = 0.0;
    for(int i=0; i<patch_size; i++){
        multReal = ref_dft_real[i] * comp_dft_real[i] - ref_dft_imag[i] * comp_dft_conj[i];
        multImag = ref_dft_real[i] * comp_dft_conj[i] + ref_dft_imag[i] * comp_dft_real[i];

        multRealAbs = fabs(multReal);
        multImagAbs = fabs(multImag);

        cross_spectrum_real[i] = ((multReal * multRealAbs) + (multImag * multImagAbs)) / ((multRealAbs * multRealAbs) +
                                 (multImagAbs * multImagAbs) + EPSILON);
        cross_spectrum_imag[i] = ((multImag * multRealAbs) - (multReal * multImagAbs)) / ((multRealAbs * multRealAbs) +
                                 (multImagAbs * multImagAbs) + EPSILON);
    }

    // Calculate normalized cross-correlation by calculating the inverse DFT of the cross-power spectrum
    double cross_corr_real[patch_size] = {0.0};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    cross_corr_real[j*bW+i] += (cross_spectrum_real[jj*bW+ii] * cos(2*PI*((1*i*ii/bW) + (1*j*jj/bH))) -
                                                cross_spectrum_imag[jj*bW+ii] * sin(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) /
                                                sqrt((double)patch_size);
                }
            }
        }
    }

    // Determine the maximum value of the cross-correlation and get peak coordinates
    double max_value = 0.0;
    //float x_coord = 0.0f;
    //float y_coord = 0.0f;
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            double value = cross_corr_real[j*bW+i];
            if(value > max_value){
                max_value = value;
                //x_coord = i; // +1 to change the index range, so that zero coordinates represent no correlation
                //y_coord = j; // +1 to change the index range, so that zero coordinates represent no correlation
            }
        }
    }

    // Calculate Euclidean distance of peak coordinates to the origin
    phase_map[gy*w+gx] = (float) max_value;
    //phase_map[gy*w+gx] = sqrt((x_coord * x_coord) + (y_coord * y_coord)); // Euclidean displacement (distance from origin)
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
