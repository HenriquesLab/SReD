//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$
#define filter_param_sq $FILTER_PARAM_SQ$
#define patch_size $PATCH_SIZE$
#define offset_x $OFFSET_X$
#define offset_y $OFFSET_Y$
float getWeight(float ref, float comp);
float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n);

kernel void kernelGetSsimMap(
    global float* ref_pixels,
    global float* local_means,
    global float* ssim_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    // Get reference patch
    float ref_patch[patch_size];
    float meanSub_x[patch_size];
    float var_x = 0;
    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = ref_pixels[j0*w+i0];
            meanSub_x[ref_counter] = ref_patch[ref_counter] - local_means[y0*w+x0];
            var_x += meanSub_x[ref_counter]*meanSub_x[ref_counter];
            ref_counter++;
        }
    }
    var_x /= patch_size;

    // For each comparison pixel...
    float weight;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

            weight = 0;

            // Get comparison patch Y
            float comp_patch[patch_size];
            float meanSub_y[patch_size];
            float var_y = 0;
            float cov_xy = 0;
            int comp_counter = 0;
            for(int j1=y1-bRH; j1<=y1+bRH; j1++){
                for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                    comp_patch[comp_counter] = ref_pixels[j1*w+i1];
                    meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[y1*w+x1];
                    var_y += meanSub_y[comp_counter]*meanSub_y[comp_counter];
                    cov_xy += meanSub_x[comp_counter]*meanSub_y[comp_counter];
                    comp_counter++;
                }
            }
            var_y /= patch_size;
            cov_xy /= patch_size;

            // Calculate weight
            weight = getWeight(local_means[y0*w+x0], local_means[y1*w+x1]);

            // Calculate RMSE(X,Y) and add it to the sum at X
            ssim_map[y0*w+x0] += getSsim(local_means[y0*w+x0], local_means[y1*w+x1], var_x, var_y, cov_xy, patch_size) * weight;
        }
    }
}

float getWeight(float mean_x, float mean_y){
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

float getSsim(float mean_x, float mean_y, float var_x, float var_y, float cov_xy, int n){
    float ssim = 0;
    float c1 = (0.01*4294967295)*(0.01*4294967295); // constant1*dynamic range
    float c2 = (0.03*4294967295)*(0.03*4294967295); // constant2*dynamic range
    float mean_x_sq = mean_x*mean_x;
    float mean_y_sq = mean_y*mean_y;

    ssim = (2*mean_x*mean_y+c1)*(2*cov_xy+c2)/((mean_x_sq+mean_y_sq+c1)*(var_x+var_y+c2));
    return ssim;
}
