/*
 * // --------------------------------- //
 * // ---- IMPORTANT - PLEASE READ ---- //
 * // --------------------------------- //
 * 
 * This macro generates an orientation weight stack. Each slice highlights regions at specific orientations.
 * 
 * The orientation map generated with the previous macro should be open before running the macro - its name should be defined in the "img" variable (e.g., "img.tif").
 * 
 * @author Afonso Mendes
 *
 */


// -------------------------- //
// ---- Define variables ---- //
// -------------------------- //

img = "image.tif";
angles = newArray(-10, -20, -30, -40, -50, -60, -70, -80, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90);


// ---------------------- //
// ---- Get skeleton ---- //
// ---------------------- //
selectWindow(img);
w=getWidth();
h=getHeight();
skeleton = newArray(w*h);
for(y=0; y<h; y++){
	for(x=0; x<w; x++){
		skeleton[y*w+x] = getPixel(x, y);
	}
}


// ------------------------- //
// ---- Generate output ---- //
// ------------------------- //

newImage("final", "32-bit black", w, h, angles.length);
for(i=1; i<=angles.length; i++){
	setSlice(i);
	for(y=0; y<h; y++){
		for(x=0; x<w; x++){
			if(skeleton[y*w+x]==angles[i-1]){
				setPixel(x, y, 1.0);				
			}
		}
	}
}
