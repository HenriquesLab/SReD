#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$

kernel void kernelGetPearsonMap(global float* refPixels,
global float* pearsonMap
){

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int bRW = bW/2;
    int bRH = bH/2;
    int patchSize = bW*bH;

    //Get reference patch pixels
    for(gy=1; gy<h-1; gy++){
        for(gx=1; gx<w-1; gx++){
            int counter = 0;
            float refPatch[patchSize];

            for(int j=gy-bRH; j<=gy+bRH; j++){
                for(int i=gx-bRW; i<=gx+bRW; i++){
                    refPixels[counter] = refPixels[j*w+i];
                    counter++;
                }
            }

        }
    }






}
