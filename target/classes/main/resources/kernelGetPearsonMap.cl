//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define filter_constant $FILTERCONSTANT$
#define EPSILON $EPSILON$

kernel void kernelGetPearsonMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* weights_sum_map,
    global float* pearson_map
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check to avoids borders dynamically based on patch dimensions
    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }

    // Check if reference pixel is an estimated noise pixel, and if so, kill the thread
    double ref_std = (double)local_stds[gy*w+gx];
    double threshold = (double)(filter_param*filter_constant);
    if((ref_std*ref_std)<threshold){
        pearson_map[gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }


    // ------------------------------------ //
    // ---- Get reference patch pixels ---- //
    // ------------------------------------ //

    double ref_patch[patch_size] = {0.0f};
    int index = 0;

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                ref_patch[index] = (double)ref_pixels[j*w+i];
                index++;
            }
        }
    }

/*
    // ----------------------------------- //
    // ---- Normalize reference patch ---- //
    // ----------------------------------- //

    // Find min and max
    double min_intensity = DBL_MAX;
    double max_intensity = -DBL_MAX;

    for(int i=0; i<patch_size; i++){
        double pixel_value = ref_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        ref_patch[i] = (ref_patch[i] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
    }

*/
    // --------------------------------------- //
    // ---- Mean-subtract reference patch ---- //
    // --------------------------------------- //

    double ref_mean = (double)local_means[gy*w+gx];

    for(int i=0; i<patch_size; i++){
        ref_patch[i] = ref_patch[i] - ref_mean;
    }

/*
    // ----------------------------------------- //
    // ---- Normalize reference patch again ---- //
    // ----------------------------------------- //

    // Find min and max
    min_intensity = DBL_MAX;
    max_intensity = -DBL_MAX;

    for(int i=0; i<patch_size; i++){
        double pixel_value = ref_patch[i];
        min_intensity = min(min_intensity, pixel_value);
        max_intensity = max(max_intensity, pixel_value);
    }

    // Remap pixel values
    for(int i=0; i<patch_size; i++){
        ref_patch[i] = (ref_patch[i] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
    }
*/

    // ------------------------------------ //
    // ---- Process comparison patches ---- //
    // ------------------------------------ //

    // Iterate over all pixels (excluding borders)
    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){


            // ------------------------------------- //
            // ---- Get comparison patch pixels ---- //
            // ------------------------------------- //

            // Check if comparison pixel is an estimated noise pixel, and if so, kill the thread
            double comp_std = (double)local_stds[y*w+x];
            if((comp_std*comp_std)<threshold){
                return;
            }

            double comp_patch[patch_size] = {0.0f};
            index = 0;
            for(int j=y-bRH; j<=y+bRH; j++){
                for(int i=x-bRW; i<=x+bRW; i++){
                    float dx = (float)(i-x);
                    float dy = (float)(j-y);
                    if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                        comp_patch[index] = (double)ref_pixels[j*w+i];
                        index++;
                    }
                }
            }

/*
            // ------------------------------------ //
            // ---- Normalize comparison patch ---- //
            // ------------------------------------ //

            // Find min and max
            min_intensity = DBL_MAX;
            max_intensity = -DBL_MAX;

            for(int k=0; k<patch_size; k++){
                double pixel_value = comp_patch[k];
                min_intensity = min(min_intensity, pixel_value);
                max_intensity = max(max_intensity, pixel_value);
            }

            // Remap pixel values
            for(int k=0; k<patch_size; k++){
                comp_patch[k] = (comp_patch[k] - min_intensity) / (max_intensity - min_intensity + (double)EPSILON);
            }
*/

            // ---------------------------------------- //
            // ---- Mean-subtract comparison patch ---- //
            // ---------------------------------------- //

            double comp_mean = (double)local_means[y*w+x];
            for(int k=0; k<patch_size; k++){
                comp_patch[k] = comp_patch[k] - comp_mean;
            }

/*
            // ------------------------------------------ //
            // ---- Normalize comparison patch again ---- //
            // ------------------------------------------ //

            // Find min and max
            min_intensity = DBL_MAX;
            max_intensity = -DBL_MAX;

            for(int k=0; k<patch_size; k++){
                double pixel_value = comp_patch[k];
                min_intensity = min(min_intensity, pixel_value);
                max_intensity = max(max_intensity, pixel_value);
            }

            // Remap pixel values
            for(int k=0; k<patch_size; k++){
                comp_patch[k] = (comp_patch[k]-min_intensity)/(max_intensity-min_intensity+(double)EPSILON);
            }
*/

            // ----------------------------------------------------- //
            // ---- Calculate Pearson's correlation coefficient ---- //
            // ----------------------------------------------------- //

            // Calculate covariance
            double covar = 0.0;
            for(int k=0; k<patch_size; k++){
                covar += ref_patch[k] * comp_patch[k];
            }
            covar /= (double)(patch_size-1);

            // Calculate Pearson's correlation coefficient
            double weight = exp((-1.0)*(((ref_std-comp_std)*(ref_std-comp_std))/((double)filter_param+(double)EPSILON)));
            weights_sum_map[gy*w+gx] += (float)weight;

            if(ref_std == 0.0 && comp_std == 0.0){
                pearson_map[gy*w+gx] += 1.0f; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
            }else if(ref_std == 0.0 || comp_std == 0.0){
                pearson_map[gy*w+gx] += 0.0f; // Special case when only one patch is flat (correlation would be NaN but we want 0 because texture has no correlation with complete lack of texture)
            }else{
                pearson_map[gy*w+gx] += (float)fmax(0.0, (covar/((ref_std*comp_std)+(double)EPSILON))) * (float)weight; // Truncate anti-correlations
            }
        }
    }
}