#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getGaussianWeight(float ref, float comp);
float getExpDecayWeight(float ref, float comp);

kernel void kernelGetPhaseCorrelationMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* phase_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(x0<offset_x || x0>=w-offset_x || y0<offset_y || y0>=h-offset_y){
        return;
    }

    int bRW = bW/2;
    int bRH = bH/2;
    double PI = 3.14159265358979323846f;
    double EPSILON = 0.00001f;

    // Get reference patch and subtract the mean
    double ref_patch[patch_size] = {0.0f};
    double meanSub_x[patch_size] = {0.0f};

    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = (double) ref_pixels[j0*w+i0];
            meanSub_x[ref_counter] = (ref_patch[ref_counter] - (double) local_means[y0*w+x0]) / 0.0001; // division by  to scale
            ref_counter++;
        }
    }

    // Calculate 2D Discrete Fourier Transform
    double ref_dft_real[patch_size] = {0};
    double ref_dft_imag[patch_size] = {0};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            for(int jj=0; jj<bH; jj++){
                for(int ii=0; ii<bW; ii++){
                    ref_dft_real[j*bW+i] += (meanSub_x[jj*bW+ii] * cos(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) / (sqrt((double)patch_size));
                    ref_dft_imag[j*bW+i] -= (meanSub_x[jj*bW+ii] * sin(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) / (sqrt((double)patch_size));
                }
            }
        }
    }

    // For each comparison pixel...
    double weight = 0.0f;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

            weight = 0.0f;

            // Get comparison patch Y
            double comp_patch[patch_size] = {0.0f};
            double meanSub_y[patch_size] = {0.0f};

            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = (double) ref_pixels[j1*w+i1];
                    meanSub_y[comp_counter] = (comp_patch[comp_counter] - (double) local_means[y1*w+x1]) / 0.0001; // Division by 0.5 to scale up the values
                    comp_counter++;
                }
            }

            // Calculate 2D Discrete Fourier Transform
            double comp_dft_real[patch_size] = {0.0f};
            double comp_dft_imag[patch_size] = {0.0f};

            for(int j=0; j<bH; j++){
                for(int i=0; i<bW; i++){
                    for(int jj=0; jj<bH; jj++){
                        for(int ii=0; ii<bW; ii++){
                            comp_dft_real[j*bW+i] += (meanSub_y[jj*bW+ii] * cos(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) / sqrt((double)patch_size);
                            comp_dft_imag[j*bW+i] -= (meanSub_y[jj*bW+ii] * sin(2*PI*((1*i*ii/bW) + (1*j*jj/bH)))) / sqrt((double)patch_size); // TODO:lots of zeroes and negative decimals but seems OK, might wanna use doubles
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

                multRealAbs = fabs(multReal);
                multImagAbs = fabs(multImag);

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
            weight = (double) getExpDecayWeight(local_stds[y0*w+x0], local_stds[y1*w+x1]);

            // Calculate Euclidean distance of peak coordinates to the origin
            phase_map[y0*w+x0] += (float) sqrt(((double) x_coord * (double) x_coord) + ((double) y_coord * (double) y_coord)) * weight; // Euclidean displacement (distance from origin)
        }
    }
}

float getGaussianWeight(float mean_x, float mean_y){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions
    // Alternative: exponential decay function: 1-abs(mean_x-mean_y/abs(mean_x+abs(mean_y)))

    float weight = 0;
    weight = mean_y - mean_x;
    weight = fabs(weight);
    weight = weight*weight;
    weight = weight/filter_param_sq;
    weight = (-1) * weight;
    weight = exp(weight);
    return weight;
}

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
