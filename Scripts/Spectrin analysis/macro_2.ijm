/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro generates rotated and zero-padded copies of the input image. 
 * 
 * The input image must be open in ImageJ - the "input" variable must be defined using the name of the image and its extension (e.g., "input.tif").
 * 
 * The reference blocks' stack should be open in ImageJ before running - the "blockStack" variable must be defined using the name of the file and its extension (e.g., "blocks.tif").
 * 
 * The path where the macro stores the results must be defined in the "outputDir" variable. The path provided should not end with a file separator (e.g., "/" or "\"), otherwise saving will fail.
 * 
 * @author Afonso Mendes
 *
 */


// -------------------------- //
// ---- Define variables ---- //
// -------------------------- //

input = "input_normalize.tif"
blockStack = "line_stack.tif"
selectWindow(blockStack);
blockW = getWidth(); // Block width
blockH = getHeight(); // Block height
blockRW = round(blockW/2); // Block radius width
blockRH = round(blockH/2); // Block radius height

outputDir = File.separator + "C:/Users/Actions/Desktop/AFONSO/spectrin_test/input_padded";


// --------------------------------- //
// ---- Define angles to rotate ---- //
// --------------------------------- //
selectWindow(blockStack);
nAngles = nSlices;
angle_step = 180/nAngles;

angles = newArray(nAngles); // List of angles to rotate each frame
angle = 90; // If first patch is vertical

for(i=0; i<nAngles; i++){
	angles[i] = angle;
	angle -= angle_step;
}


// ------------------------- //
// ---- Get input image ---- //
// ------------------------- //

selectWindow(input);
w=getWidth();
h=getHeight();
input = newArray(w*h);
for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		input[y*w+x] = getPixel(x, y);
	}
}

// ------------------------------ //
// ---- Get final dimensions ---- //
// ------------------------------ //

diagonal = sqrt(w*w + h*h);
paddingW = Math.ceil((diagonal-w+1)+blockRW);
paddingH = Math.ceil((diagonal-h+1)+blockRH);
newW = w+paddingW;
newH = h+paddingH;

// ------------------------------ //
// ---- Build rotated inputs ---- //
// ------------------------------ //

for(i=0; i<nAngles; i++){
	// Build temporary input image
	newImage("input_rotated_"+angles[i], "32-bit black", w, h, 1);
	for(y=0; y<h; y++){
		for(x=0; x<w; x++){
			setPixel(x, y, input[y*w+x]);
		}
	}
	tempTitle=getTitle();
	
	// Add padding (to not lose information outside bounds when rotating
	run("Canvas Size...", "width=" + newW + " height="+ newH +" position=Center zero");
	
	// Rotate
	run("Rotate... ", "angle="+ angles[i] +" grid=1 interpolation=Bicubic");
	
	// Save
	saveAs("TIFF", outputDir+ File.separator + "input_padded_rotated_"+angles[i]+".tif");
	
	close("input_padded_rotated_"+angles[i]+".tif");
}

print("Done");
