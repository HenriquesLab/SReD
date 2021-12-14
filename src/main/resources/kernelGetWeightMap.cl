#define w $WIDTH$
#define h $HEIGHT$

kernel void kernelGetWeightMap(
    global float* localMeans,
    global float* weightMap
){
    // Calculate weight (based on the Gaussian weight function used in non-local means
    // (see https://en.wikipedia.org/wiki/Non-local_means#Common_weighting_functions)
    // TODO: Check division by zero - also the function is missing the filtering parameter
    // Java expression: exp((-1)*pow(abs(patchStats1[0]-patchStats0[0]),2)/pow(0.4F*sigma,2))
    // Can also try exponential decay function: 1-abs(patchStats0[0]-patchStats1[0]/abs(patchStats0[0]+abs(patchStats1[0])))

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    float weight = 0;

    for(gy=1; gy<h-1; gy++){
        for(gx=1; gx<w-1; gx++){
            for(int y=1; y<h-1; y++){
                for(int x=1; x<w-1; x++){
                    weight = localMeans[y*w+x] - localMeans[gy*w+gx];
                    weight = fabs(weight);
                    weight = weight*weight;
                    weight = (-1) * weight;
                    weight = exp(weight);
                    //weight = weight/filteringParamSq;
                    weightMap[gy*w+gx] = weight;
                }
            }
        }
    }
}