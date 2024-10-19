//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define image_width $WIDTH$
#define image_height $HEIGHT$
#define image_depth $DEPTH$
#define block_size $BLOCK_SIZE$
#define bRW $BRW$
#define bRH $BRH$
#define bRZ $BRZ$
#define filter_param $FILTER_PARAM$
#define threshold $THRESHOLD$
#define EPSILON $EPSILON$

kernel void kernelGetGlobalNrmse3D(
    global float* ref_pixels,
    global float* local_means,
    global float* local_stds,
    global float* weights_sum_map,
    global float* rmse_map
){

    // Get global indexes
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gz = get_global_id(2);

    // Bound check to avoid borders
    if(gx<bRW || gx>=image_width-bRW || gy<bRH || gy>=image_height-bRH || gz<bRZ || gz>=image_depth-bRZ){
        rmse_map[image_width*image_height*gz+gy*image_width+gx] = 0.0f;
        return;
    }

    // Check to avoid blocks with no structural relevance
    int ref_index = image_width*image_height*gz+gy*image_width+gx;
    float ref_std = local_stds[ref_index];
    float ref_var = ref_std*ref_std;

    if(ref_var<threshold || ref_var==0.0f){
        rmse_map[ref_index] = 0.0f; // Set pixel to zero to avoid retaining spurious values already in memory
        return;
    }

    // Get reference block
    float ref_block[block_size] = {0.0f};
    int index = 0;

    for(int z=gz-bRZ; z<gz+bRZ; z++){
        for(int y=gy-bRH; y<=gy+bRH; y++){
            for(int x=gx-bRW; x<=gx+bRW; x++){
                float dx = (float)(x-gx);
                float dy = (float)(y-gy);
                float dz = (float)(z-gz);
                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                    ref_block[index] = ref_pixels[image_width*image_height*z+y*image_width+x];
                    index++;
                }
            }
        }
    }

    // Mean-subtract reference block
    float ref_mean = local_means[ref_index];
    for(int i=0; i<block_size; i++){
        ref_block[i] = ref_block[i] - ref_mean;
    }

    // Calculate NRMSE
    for(int z=bRZ; z<image_depth-bRZ; z++){
        for(int y=bRH; y<image_height-bRH; y++){
            for(int x=bRW; x<image_width-bRW; x++){

                // Get test block pixels, checking to avoid blocks with no structural relevance
                float test_std = local_stds[image_width*image_height*z+y*image_width+x];
                float test_var = (float)test_std*(float)test_std;

                if(test_var<threshold){
                    rmse_map[ref_index] += 0.0f;
                }else{
                    float test_block[block_size] = {0.0f};
                    index = 0;
                    for(int zz=z-bRZ; zz<=z+bRZ; zz++){
                        for(int yy=y-bRH; yy<=y+bRH; yy++){
                            for(int xx=x-bRW; xx<=x+bRW; xx++){
                                float dx = (float)(xx-x);
                                float dy = (float)(yy-y);
                                float dz = (float)(zz-z);
                                if(((dx*dx)/(float)(bRW*bRW))+((dy*dy)/(float)(bRH*bRH))+((dz*dz)/(float)(bRZ*bRZ)) <= 1.0f){
                                    test_block[index] = ref_pixels[image_width*image_height*zz+yy*image_width+xx];
                                    index++;
                                }
                            }
                        }
                    }

                    // Mean-subtract test block
                    float test_mean = local_means[image_width*image_height*z+y*image_width+x];
                    for(int i=0; i<block_size; i++){
                        test_block[i] = test_block[i] - test_mean;
                    }

                    // Calculate weight
                    float weight = (float)exp((float)(-1.0f)*(float)(((ref_std-test_std)*(ref_std-test_std))/(filter_param+EPSILON)));
                    weights_sum_map[ref_index] += weight;

                    // Calculate NRMSE
                    float mse = 0.0f;
                    for(int i=0; i<block_size; i++){
                        mse += (ref_block[i]-test_block[i])*(ref_block[i]-test_block[i]);
                    }
                    mse /= (float) block_size;

                    rmse_map[ref_index] += sqrt(mse)*weight;
                }
            }
        }
    }
}