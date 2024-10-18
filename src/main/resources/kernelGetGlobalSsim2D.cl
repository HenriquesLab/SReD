//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define block_size $BLOCK_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define filter_param $FILTER_PARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetGlobalSsim2D(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* weights_sum_map,
    global float* ssim_map
){

    // Get global indexes
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Bound check to avoid borders
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH){
        return;
    }

    // Check to avoid blocks with no structural relevance
    float ref_std = local_stds[gy*image_width+gx];
    float ref_var = (float)ref_std*(float)ref_std;

    if(ref_var<threshold){
        ssim_map[gy*image_width+gx] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Get reference block
    float ref_block[block_size] = {0.0f};
    int index = 0;

    for(int y=gy-bRH; y<=gy+bRH; y++){
        for(int x=gx-bRW; x<=gx+bRW; x++){
            float dx = (float)(x-gx);
            float dy = (float)(y-gy);
            if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                ref_block[index] = ref_pixels[y*image_width+x];
                index++;
            }
        }
    }

    // Mean-subtract reference block
    float ref_mean = local_means[gy*image_width+gx];
    for(int i=0; i<block_size; i++){
        ref_block[i] = ref_block[i] - ref_mean;
    }

    // Calculate similarity between the reference and test blocks
    for(int y=bRH; y<image_height-bRH; y++){
        for(int x=bRW; x<image_width-bRW; x++){


            // Get test block pixels, checking to avoid blocks with no structural relevance
            float test_std = local_stds[y*image_width+x];
            float test_var = (float)test_std*(float)test_std;

            if(test_var<threshold){
                ssim_map[gy*image_width+gx] += 0.0f;
            }else{
                float test_block[block_size] = {0.0f};
                index = 0;
                for(int yy=y-bRH; yy<=y+bRH; yy++){
                    for(int xx=x-bRW; xx<=x+bRW; xx++){
                        float dx = (float)(xx-x);
                        float dy = (float)(yy-y);
                        if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH)) <= 1.0f){
                            test_block[index] = ref_pixels[yy*image_width+xx];
                            index++;
                        }
                    }
                }


                // Mean-subtract test block
                float test_mean = local_means[y*image_width+x];
                for(int i=0; i<block_size; i++){
                    test_block[i] = test_block[i] - test_mean;
                }

                // Calculate covariance
                float covariance = 0.0f;
                for(int i=0; i<block_size; i++){
                    covariance += ref_block[i] * test_block[i];
                }
                covariance /= (float)(block_size-1);


                // Calculate weight
                float weight = (float)exp((float)(-1.0f)*(float)(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                weights_sum_map[gy*image_width+gx] += weight;

                // Calculate SSIM
                //float c1 = (0.01f * 1.0f) * (0.01f * 1.0f);
                float c1 = 0.0001f;
                //float c2 = (0.03f * 1.0f) * (0.03f * 1.0f);
                float c2 = 0.0009f;
                //float c3 = c2/2.0f;
                float c3 = 0.00045f;

                if(ref_std==0.0f && test_std==0.0f){
                    ssim_map[gy*image_width+gx] += 1.0f * (float)weight; // Special case when both block are flat, correlation is 1
                }else if(ref_std==0.0f || test_std==0.0f){
                    ssim_map[gy*image_width+gx] += 0.0f; // Special case when one block is flat, correlation is 0 (weight here is irrelevant)
                }else{
                    float ssim = ((2.0f*ref_mean*test_mean+c1)*(2.0f*covariance+c2))/(((ref_mean*ref_mean)*(test_mean*test_mean)+c1)*((ref_std*ref_std)+(test_std*test_std)+c2))*weight;
                    ssim_map[gy*image_width+gx] += (float) fmax(0.0f, ssim);
                }
            }
        }
    }
}