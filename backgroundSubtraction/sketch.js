// ********************************
/**
 * Object Detection with Background Subtraction
 * Adopted from Graphics Programming Course at Goldsmiths, University of London.
 */
// ********************************
var video;
var backImg; // background image
var diffImg; // difference between background and current images, for debugging
var currImg; // current image. No need for this variable. You can use video instead
var thresholdSlider; // threshold slider
var threshold; // threshold for the difference

function setup()
{
    createCanvas(640 * 2, 480);
    pixelDensity(1);
    video = createCapture(VIDEO); // capture video
    video.hide();

    thresholdSlider = createSlider(0, 255, 50);
    thresholdSlider.position(20, 20);
}

function draw()
{
    background(0);
    image(video, 0, 0); // display video

    // create a new image that has the same dimensions as the video
    currImg = createImage(video.width, video.height);
    // copy the entire video image into the current image.
    currImg.copy(video, 0, 0, video.width, video.height, 0, 0, video.width, video.height);

    // create a new image that has the same dimensions as the video
    diffImg = createImage(video.width, video.height);
    diffImg.loadPixels(); // get ready to edit that image

    threshold = thresholdSlider.value(); // set the threshold

    // Only run this block when a background image is created (i.e. a key is pressed)
    if (typeof backImg !== 'undefined')
    {
        backImg.loadPixels(); // load BackImg pixels to access them
        currImg.loadPixels(); // load currImg pixels to access them

        // loop over every pixel in the current image
        for (var x = 0; x < currImg.width; x++)
        {
            for (var y = 0; y < currImg.height; y++)
            {
                var index = (x + (y * currImg.width)) * 4; // calculate pixel's index
                var redSource = currImg.pixels[index + 0]; // get pixel's red channel from currImg
                var greenSource = currImg.pixels[index + 1]; // get pixel's green channel from currImg
                var blueSource = currImg.pixels[index + 2]; // get pixel's blue channel from currImg

                var redBack = backImg.pixels[index + 0]; // get pixel's red channel from backImg
                var greenBack = backImg.pixels[index + 1]; // get pixel's green channel from backImg
                var blueBack = backImg.pixels[index + 2]; // get pixel's blue channel from backImg

                // calculate the distance between the background pxl and the current pxl
                var d = dist(redSource, greenSource, blueSource, redBack, greenBack, blueBack);

                // if the distance is greater than the threshold, make the pixels black to discern
                // the object from the background
                if (d > threshold)
                {
                    diffImg.pixels[index + 0] = 0;
                    diffImg.pixels[index + 1] = 0;
                    diffImg.pixels[index + 2] = 0;
                    diffImg.pixels[index + 3] = 255;
                } else { // else make the pixel white.
                    diffImg.pixels[index + 0] = 255;
                    diffImg.pixels[index + 1] = 255;
                    diffImg.pixels[index + 2] = 255;
                    diffImg.pixels[index + 3] = 255;
                }
            }
        }
    }

    diffImg.updatePixels(); // update the diffImg since we are changing its pixels
    image(diffImg, 640, 0); // show the diffImg next to the current image

    noFill();
    stroke(255);
    text(threshold, 160, 35);
}

function keyPressed()
{
    if (keyCode == 32)
    {
        // When a key is pressed, create a background image that is the same size as the current image
        backImg = createImage(currImg.width, currImg.height);
        // Copy the entire current image into the background image
        backImg.copy(currImg, 0, 0, currImg.width, currImg.height, 0, 0, currImg.width, currImg.height);
        console.log("saved new background");
    }
}

// // faster method for calculating color similarity which does not calculate root.
// // Only needed if dist() runs slow
// function distSquared(x1, y1, z1, x2, y2, z2)
// {
//   var d = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
//   return d;
// }
