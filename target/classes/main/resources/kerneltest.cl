#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define ref_mean $MEAN_X$
#define std_x $STD_X$
#define ref_var $VAR_X$
#define hu_x $HU_X$

float getExpDecayWeight(float ref, float comp);
float getNrmse(float* ref, float* comp, float mean_y, int n);
float getMae(float* ref, float* comp, int n);
float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n);
float getInvariant(float* patch, int patch_w, int patch_h, int p, int q);

kernel void kernelGetLocalMeans(
global float* ref_pixels,
global float* local_means,
global float* local_stds
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get patch and calculate local mean
    float mean = 0.0f;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            mean += ref_pixels[j*w+i];
        }
    }
    local_means[gy*w+gx] = mean/patch_size;

    // Calculate standard deviation
    float std = 0.0f;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            std += (ref_pixels[j*w+i]-local_means[gy*w+gx]) * (ref_pixels[j*w+i]-local_means[gy*w+gx]);
        }
    }
    local_stds[gy*w+gx] = sqrt(std/patch_size);
}

kernel void kernelGetPearsonMap(
    global float* ref_patch,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* pearson_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // For each comparison pixel...
    float weight = 0.0f;

    // Get patch minimum and maximum
    float min_y = ref_pixels[gy*w+gx];
    float max_y = ref_pixels[gy*w+gx];

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            if(ref_pixels[j*w+i] < min_y){
                min_y = ref_pixels[j*w+i];
            }
            if(ref_pixels[j*w+i] > max_y){
                max_y = ref_pixels[j*w+i];
            }
        }
    }

    // Get values, subtract the mean, and get local standard deviation
    float comp_patch[patch_size] = {0.0f};
    float meanSub_y[patch_size] = {0.0f};
    float std_y = local_stds[gy*w+gx];
    float meanSub_xy = 0.0f;

    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = ref_pixels[j*w+i];
            //comp_patch[comp_counter] = (ref_pixels[j*w+i] - min_y) / (max_y - min_y + 0.00001f); // Normalize patch to [0,1]
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
            meanSub_xy += ref_patch[comp_counter] * meanSub_y[comp_counter];
            comp_counter++;
        }
    }

    // Calculate weight
    weight = getExpDecayWeight(std_x, std_y);

    // Calculate Pearson correlation coefficient X,Y and add it to the sum at X (avoiding division by zero)
    if(std_x == 0.0f && std_y == 0.0f){
        pearson_map[gy*w+gx] = 1.0f * weight; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
    }else{
        pearson_map[gy*w+gx] = max(0.0f, (meanSub_xy / ((std_x * std_y) + 0.00001f)) * weight); // Truncate anti-correlations to zero
    }
}

kernel void kernelGetNrmseMap(
    global float* ref_patch,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* nrmse_map,
    global float* mae_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Get reference patch
    float refPatch[patch_size] = {0.0f};
    for(int j=0; j<bH; j++){
        for(int i=0; i<bW; i++){
            refPatch[j*bW+i] = ref_patch[j*bW+i];
        }
    }
    // For each comparison pixel...
    float weight = 0.0f;

    // Get comparison patch minimum and maximum
    float min_y = ref_pixels[gy*w+gx];
    float max_y = ref_pixels[gy*w+gx];

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            if(ref_pixels[j*w+i] < min_y){
               min_y = ref_pixels[j*w+i];
            }
            if(ref_pixels[j*w+i] > max_y){
                max_y = ref_pixels[j*w+i];
            }
        }
    }

    // Get comparison patch Y
    float comp_patch[patch_size];
    float meanSub_y[patch_size];
    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = ref_pixels[j*w+i];
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
            comp_counter++;
        }
    }

    // Calculate weight
    weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

    // Calculate NRMSE(X,Y) and add it to the sum at X
    nrmse_map[gy*w+gx] = getNrmse(refPatch, meanSub_y, local_means[gy*w+gx], patch_size) * weight;
    mae_map[gy*w+gx] = getMae(refPatch, meanSub_y, patch_size) * weight;
}


kernel void kernelGetSsimMap(
    global float* ref_patch,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* ssim_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // For each comparison pixel...
    float weight = 0.0f;

    // Get comparison patch Y
    float comp_patch[patch_size] = {0.0f};
    float meanSub_y[patch_size] = {0.0f};
    float var_y = 0.0f;
    float cov_xy = 0.0f;
    int comp_counter = 0;
    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            comp_patch[comp_counter] = ref_pixels[j*w+i];
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
            cov_xy += ref_patch[comp_counter]*meanSub_y[comp_counter];
            comp_counter++;
        }
    }
    var_y = local_stds[gy*w+gx] * local_stds[gy*w+gx];
    cov_xy /= patch_size;

    // Calculate weight
    weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

    // Calculate SSIM and add it to the sum at X
    ssim_map[gy*w+gx] = getSsim(ref_mean, local_means[gy*w+gx], ref_var, var_y, cov_xy, patch_size) * weight;
}

kernel void kernelGetHuMap(
    global float* ref_patch,
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* hu_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check (avoids borders dynamically based on patch dimensions)
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // For each comparison pixel...
    float weight = 0.0f;
    float invariant_20_y = 0.0f;
    float invariant_02_y = 0.0f;
    float hu_y = 0.0f;

    for(int gy=bRH; gy<h-bRH; gy++){
        for(int gx=bRW; gx<w-bRW; gx++){

            weight = 0.0f;
            invariant_20_y = 0.0f;
            invariant_02_y = 0.0f;
            hu_y = 0.0f;

            // Get comparison patch Y
            float comp_patch[patch_size] = {0.0f};
            float meanSub_y[patch_size] = {0.0f};
            int comp_counter = 0;

            for(int j=gy-bRH; j<=gy+bRH; j++){
                for(int i=gx-bRW; i<=gx+bRW; i++){
                    comp_patch[comp_counter] = ref_pixels[j*w+i];
                    //comp_patch[comp_counter] = (ref_pixels[j*w+i] - min_y) / (max_y - min_y + 0.00001f); // Normalize patch to [0,1]
                    meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[gy*w+gx];
                    comp_counter++;
                }
            }

            // Calculate Hu moment 2 for comparison patch
            invariant_20_y = getInvariant(meanSub_y, bW, bH, 2, 0);
            invariant_02_y = getInvariant(meanSub_y, bW, bH, 0, 2);
            hu_y = invariant_20_y + invariant_02_y;

            // Calculate weight
            weight = getExpDecayWeight(std_x, local_stds[gy*w+gx]);

            // Calculate Euclidean distance between Hu moments and add to Hu map
            hu_map[gy*w+gx] = fabs((float) hu_y - (float) hu_x) * weight;
        }
    }
}

kernel void kernelGetPhaseCorrelationMap(
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
    phase_map[gy*w+gx] = (float) sqrt(((double) x_coord * (double) x_coord) + ((double) y_coord * (double) y_coord)) * weight; // Euclidean displacement (distance from origin)
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

float getNrmse(float* ref, float* comp, float mean_y, int n){
    float foo = 0;
    float nrmse = 0;
    for(int i=0; i<n; i++){
        foo = ref[i] - comp[i];
        foo = foo*foo;
        nrmse += foo;
    }
    nrmse = nrmse/n;
    nrmse = sqrt(nrmse);
    nrmse = nrmse/(mean_y+0.00001f);
    nrmse = nrmse;

    return nrmse;
}

float getMae(float* ref, float* comp, int n){
    float foo = 0;
    float mae = 0;
    for(int i=0; i<n; i++){
        foo = ref[i] - comp[i];
        foo = fabs(foo);
        mae += foo;
    }
    mae = mae/n;
    return mae;
}

float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n){
    float ssim = 0;
    float c1 = (0.01*4294967295)*(0.01*4294967295); // constant1*dynamic range
    float c2 = (0.03*4294967295)*(0.03*4294967295); // constant2*dynamic range
    float mean_x_sq = mean_x*mean_x;
    float mean_y_sq = mean_y*mean_y;

    ssim = (2*mean_x*mean_y+c1)*(2*cov_xy+c2)/((mean_x_sq+mean_y_sq+c1)*(var_x+var_y+c2));
    return ssim;
}

float getInvariant(float* patch, int patch_w, int patch_h, int p, int q){
    float moment_10 = 0.0f;
    float moment_01 = 0.0f;
    float moment_00 = 0.0f;
    float centroid_x = 0.0f;
    float centroid_y = 0.0f;
    float mu_pq = 0.0f;
    float invariant = 0.0f;

    // Get centroids x and y
    for(int j=0; j<patch_h; j++){
        for(int i=0; i<patch_w; i++){
            moment_10 += patch[j*patch_w+i] * pown((float) i+1, (int) 1);
            moment_01 += patch[j*patch_w+i] * pown((float) j+1, (int) 1);
            moment_00 += patch[j*patch_w+i];
        }
    }

    // Avoid division by zero
    if(moment_00 < 0.00001f){
        moment_00 += 0.00001f;
    }

    centroid_x = moment_10/moment_00;
    centroid_y = moment_01/moment_00;

    for(int j=0; j<patch_h; j++){
            for(int i=0; i<patch_w; i++){
                mu_pq += patch[j*patch_w+i] * pown((float) i+1-centroid_x, (int) p) * pown((float) j+1-centroid_y, (int) q);
            }
    }

    invariant = mu_pq / pow(moment_00, (float) (1+(p+q/2)));
    return invariant;
}

