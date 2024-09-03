/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro generates an orientation map where each region of a skeleton is labelled with its corresponding angle. NOTE: Pixels outside the skeleton will be labelled "-2000" to avoid confusion with angles of 0.
 * 
 * The original skeleton must be open before running the macro - its name should be defined in the "input" variable (e.g., "skeleton.tif").
 * 
 * The orientation maps calculated with SReD must be open as a stack - its name should be defined in the "stack" variable (e.g., "stack.tif").
 * 
 * @author Afonso Mendes
 *
 */


// -------------------------- //
// ---- Define variables ---- //
// -------------------------- //

input = "skeleton.tif";
stack = "sred.tif";


// ---------------------- //
// ---- Get skeleton ---- //
// ---------------------- //

selectWindow(input);
w=getWidth();
h=getHeight();
skeleton = newArray(w*h);
for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		skeleton[y*w+x] = getPixel(x, y);
	}
}


// ------------------------------------------------------------ //
// ---- Get max of each skeleton pixel across orientations ---- //
// ------------------------------------------------------------ //

angles = newArray(-10, -20, -30, -40, -50, -60, -70, -80, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90);
selectWindow(stack);
n = nSlices;
output = newArray(w*h);
for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		if(skeleton[y*w+x]==255){
			max = 0.0;
			index = 1;
			for(i=1; i<=n; i++){
				setSlice(i);
				pixel = getPixel(x, y);
				if(pixel>max){
					max = pixel;
					index = i;
				}
			}
			output[y*w+x] = angles[index-1];
		}else{
			output[y*w+x] = -2000;
		}
	}
}


// ------------------------- //
// ---- Generate output ---- //
// ------------------------- //
newImage("output", "32-bit black", w, h, 1);

for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		setPixel(x, y, output[y*w+x]);
	
	}
}		
		
		
		
		
		
