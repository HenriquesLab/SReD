//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

kernel void kernelGetPearsonMap(
    global float* ref_pixels,
    global float* local_means,
    global float* pearson_map
){

    int x0 = get_global_id(0);
    int y0 = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;

    // ---- Get reference patch ----
    // Get reference patch minimum and maximum
    float min_x = ref_pixels[y0*w+x0];
    float max_x = ref_pixels[y0*w+x0];

    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            if(ref_pixels[j0*w+i0] < min_x){
                min_x = ref_pixels[j0*w+i0];
            }
            if(ref_pixels[j0*w+i0] > max_x){
                max_x = ref_pixels[j0*w+i0];
            }
        }
    }

    // Get reference patch values, subtract the mean, and get local standard deviation
    float ref_patch[patch_size];
    float meanSub_x[patch_size];
    float std_x = 0;
    int ref_counter = 0;
    for(int j0=y0-bRH; j0<=y0+bRH; j0++){
        for(int i0=x0-bRW; i0<=x0+bRW; i0++){
            ref_patch[ref_counter] = (ref_pixels[j0*w+i0] - min_x) / (max_x - min_x + 0.000001f); // Normalize patch to [0,1]
            meanSub_x[ref_counter] = ref_patch[ref_counter] - local_means[y0*w+x0];
            std_x += meanSub_x[ref_counter]*meanSub_x[ref_counter];
            ref_counter++;
        }
    }
    std_x = sqrt(std_x);

    // For each comparison pixel...
    float weight;
    for(int y1=offset_y; y1<h-offset_y; y1++){
        for(int x1=offset_x; x1<w-offset_x; x1++){

        weight = 0;

        // Get comparison patch minimum and maximum
        float min_y = ref_pixels[y1*w+x1];
        float max_y = ref_pixels[y1*w+x1];

        for(int j1=y1-bRH; j1<=y1+bRH; j1++){
            for(int i1=x1-bRW; i1<=x1+bRW; i1++){
                if(ref_pixels[j1*w+i1] < min_y){
                    min_y = ref_pixels[j1*w+i1];
                }
                if(ref_pixels[j1*w+i1] > max_y){
                    max_y = ref_pixels[j1*w+i1];
                }
            }
        }

        // Get comparison patch values, subtract the mean, and get local standard deviation
        float comp_patch[patch_size];
        float meanSub_y[patch_size];
        float std_y = 0;
        float meanSub_xy = 0;
        int comp_counter = 0;

        for(int j1=y1-bRH; j1<=y1+bRH; j1++){
            for(int i1=x1-bRW; i1<=x1+bRW; i1++){
            comp_patch[comp_counter] = (ref_pixels[j1*w+i1] - min_y) / (max_y - min_y + 0.000001f); // Normalize patch to [0,1]
            meanSub_y[comp_counter] = comp_patch[comp_counter] - local_means[y1*w+x1];
            std_y += meanSub_y[comp_counter] * meanSub_y[comp_counter];
            meanSub_xy += meanSub_x[comp_counter] * meanSub_y[comp_counter];
            comp_counter++;
            }
        }
        std_y = sqrt(std_y);

        // Calculate weight
        weight = getGaussianWeight(std_x, std_y);

        // Calculate Pearson correlation coefficient X,Y and add it to the sum at X (avoiding division by zero)
        pearson_map[y0*w+x0] += (1-(meanSub_xy/(std_x*std_y)+0.000001f)) * weight; // Pearson distance
        //pearson_map[y0*w+x0] += sqrt(1-(meanSub_xy/((std_x*std_y)*(std_x*std_y))+0.000001f)) * weight; // srqt P^earson distance
        //pearson_map[y0*w+x0] += fmax((float) 0.0f, meanSub_xy/((std_x*std_y)+0.000001f)) * weight; // truncated pearson corr
        }
    }
}

float getGaussianWeight(float ref, float comp){
    // Gaussian weight, see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions

    float weight = 0;
    weight = comp - ref;
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


    weight = 1-(fabs(ref-comp)/fabs(ref+fabs(comp)));
    return weight;

}