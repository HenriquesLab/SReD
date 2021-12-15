#define w $WIDTH$
#define h $HEIGHT$
#define bW $BW$
#define bH $BH$

kernel void kernelGetPearsonMap(

global float* refPixels,
global float* pearsonMap

){

    int bRW = bW/2; // half of the block width
    int bRH = bH/2; // half of the block height

    // Get reference patch pixels
    float refPatch[bW*bH];
    int refCounter = 0;
    for(int j=1; j






}