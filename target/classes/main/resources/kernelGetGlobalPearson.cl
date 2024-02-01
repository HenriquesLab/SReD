//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define w $WIDTH$
#define h $HEIGHT$
#define patch_size $PATCH_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTERPARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetPearsonMap(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* weights_sum_map,
    global float* pearson_map
){

    // ---------------------------- //
    // ---- Get global indexes ---- //
    // ---------------------------- //

    int gx = get_global_id(0);
    int gy = get_global_id(1);


    // -------------------------------------- //
    // ---- Bound check to avoid borders ---- //
    // -------------------------------------- //

    if(gx<bRW || gx>=w-bRW || gy<bRH || gy>=h-bRH){
        return;
    }


    // ------------------------------------------------------------ //
    // ---- Check to avoid blocks with no structural relevance ---- //
    // ------------------------------------------------------------ //

    float ref_std = local_stds[gy*w+gx];
    float ref_var = (float)ref_std*(float)ref_std;

    if(ref_var<threshold || ref_var==0.0f){
        pearson_map[gy*w+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }


    // ----------------------------- //
    // ---- Get reference block ---- //
    // ----------------------------- //

    float ref_patch[patch_size] = {0.0f};
    int index = 0;

    for(int j=gy-bRH; j<=gy+bRH; j++){
        for(int i=gx-bRW; i<=gx+bRW; i++){
            float dx = (float)(i-gx);
            float dy = (float)(j-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                ref_patch[index] = ref_pixels[j*w+i];
                index++;
            }
        }
    }


    // --------------------------------------- //
    // ---- Mean-subtract reference block ---- //
    // --------------------------------------- //

    float ref_mean = local_means[gy*w+gx];
    for(int i=0; i<patch_size; i++){
        ref_patch[i] = ref_patch[i] - ref_mean;
    }


    // -------------------------------------------------------------------- //
    // ---- Calculate similarity between the reference and test blocks ---- //
    // -------------------------------------------------------------------- //

    for(int y=bRH; y<h-bRH; y++){
        for(int x=bRW; x<w-bRW; x++){


            // ------------------------------- //
            // ---- Get test block pixels ---- //
            // ------------------------------- //

            // Check to avoid blocks with no structural relevance
            float test_std = local_stds[y*w+x];
            float test_var = (float)test_std*(float)test_std;

            if(test_var<threshold || test_var==0.0f){
                pearson_map[gy*w+gx] += 0.0f;
            }else{
                float test_patch[patch_size] = {0.0f};
                index = 0;
                for(int j=y-bRH; j<=y+bRH; j++){
                    for(int i=x-bRW; i<=x+bRW; i++){
                        float dx = (float)(i-x);
                        float dy = (float)(j-y);
                        if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                            test_patch[index] = ref_pixels[j*w+i];
                            index++;
                        }
                    }
                }


                // ---------------------------------- //
                // ---- Mean-subtract test block ---- //
                // ---------------------------------- //

                float test_mean = local_means[y*w+x];
                for(int k=0; k<patch_size; k++){
                    test_patch[k] = test_patch[k] - test_mean;
                }


                // ------------------------------- //
                // ---- Calculate covariance ----- //
                // ------------------------------- //

                float covariance = 0.0;
                for(int k=0; k<patch_size; k++){
                    covariance += ref_patch[k] * test_patch[k];
                }
                covariance /= (float)(patch_size-1);


                // ----------------------------------------------------- //
                // ---- Calculate Pearson's correlation coefficient ---- //
                // ----------------------------------------------------- //

                float weight = exp((-1.0)*(((ref_std-test_std)*(ref_std-test_std))/(filter_param+(float)EPSILON)));
                weights_sum_map[gy*w+gx] += (float)weight;

                if(ref_std == 0.0 && test_std == 0.0){
                    pearson_map[gy*w+gx] += 1.0f * (float)weight; // Special case when both patches are flat (correlation would be NaN but we want 1 because textures are the same)
                }else if(ref_std == 0.0 || test_std == 0.0){
                    pearson_map[gy*w+gx] += 0.0f; // Special case when only one patch is flat (correlation would be NaN but we want 0 because texture has no correlation with complete lack of texture)
                }else{
                    pearson_map[gy*w+gx] += (float)fmax(0.0, (covariance/((ref_std*test_std)+EPSILON))) * (float)weight; // Truncate anti-correlations
                }
            }
        }
    }
}