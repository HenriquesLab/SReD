#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$

kernel void kernelGetLocalDeviations(
global float* refPixels,
global float* localDeviations
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int bRW = bW/2;
    int bRH = bH/2;
    int patchSize = bW*bH;

    for (gy=1; gy<h-1; gy++) {
        for (gx=1; gx<w-1; gx++) {

            float sum = 0;
            float sq_sum = 0;
            for(int j=gy-bRH; j<=gy+bRH; j++){
                for(int i=gx-bRW; i<=gx+bRW; i++){
                    sum += refPixels[j*w+i];
                    sq_sum += refPixels[j*w+i]*refPixels[j*w+i];
                }
            }
            float mean = sum/patchSize;
            float variance = sq_sum/patchSize - mean*mean;
            localDeviations[gy*w+gx] = sqrt(variance);
        }
    }


}
