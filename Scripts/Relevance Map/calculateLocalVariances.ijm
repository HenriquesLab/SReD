/**
 * 
 * This macro calculates the local variances in an image using non-overlapping blocks.
 * 
 * The input image must be open in ImageJ before running the macro - the "imgTitle" variable should be defined using the name of the image (e.g., "input.tif").
 * 
 * @author Afonso Mendes
 * 
 */


// Define variables
imgTitle = "input.tif";
w=getWidth();
h=getHeight();
wh=w*h;
CIF = 352*288; // Resolution of a CIF file, used as reference for block size

// Get image array
selectWindow(imgTitle);
img = newArray(wh);
for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		img[y*w+x]=getPixel(x, y);
	}
}

// Define block size
blockLength = 0; // Length of the block
if(wh<=CIF){
	blockLength = 8;
}else{
	blockLength = 16;
}

blockSize = blockLength*blockLength; // Block size/area

// Calculate number of blocks
nBlocksX = floor(w/blockLength); // Number of blocks in X
nBlocksY = floor(h/blockLength); // Number of blocks in Y
nBlocks = floor(nBlocksX*nBlocksY); // Total number of blocks

// Calculate local variances (non-overlapping blocks)
localVars = newArray(nBlocks); // Array to store local variances
varIndex = 0;
for(y=0; y<nBlocksY; y++){
	for(x=0; x<nBlocksX; x++){
		// Get block
		tempBlock = newArray(blockSize);
		tempIndex = 0;
		for(yy=y*blockLength; yy<(y+1)*blockLength; yy++){
			for(xx=x*blockLength; xx<(x+1)*blockLength; xx++){
				//makeRectangle(xx, yy, blockLength, blockLength);
				tempBlock[tempIndex] = getPixel(xx, yy);
				tempIndex++;
			}
		}
		
		// Calculate block variance
		Array.getStatistics(tempBlock, min, max, mean, stdDev);
		localVars[varIndex]=stdDev*stdDev;
		varIndex++;
	}
}
Array.show(localVars); // Save this array as CSV for further processing



