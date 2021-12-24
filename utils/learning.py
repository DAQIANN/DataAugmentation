import cv2
import numpy as np

def cropping():
    # link to tutorial https://learnopencv.com/cropping-an-image-using-opencv/

    img = cv2.imread('/Users/akhilkodumuri/Desktop/image.img')
    print(img.shape) # Print image shape
    cv2.imshow("original", img)
    cv2.waitKey(1)

    # Cropping an image
    cropped_image = img[80:280, 150:330]

    print("cropping image")
    # Display cropped image
    cv2.imshow("cropped", cropped_image)
    cv2.waitKey(1)

    # Save the cropped image
    # cv2.imwrite("Cropped Image.jpg", cropped_image)

    cv2.destroyAllWindows()

    img =  cv2.imread("test_cropped.jpg")
    image_copy = img.copy() 
    imgheight=img.shape[0]
    imgwidth=img.shape[1]
    M = 76
    N = 104
    x1 = 0
    y1 = 0

    for y in range(0, imgheight, M):
        for x in range(0, imgwidth, N):
            if (imgheight - y) < M or (imgwidth - x) < N:
                break
                
            y1 = y + M
            x1 = x + N

            # check whether the patch width or height exceeds the image width or height
            if x1 >= imgwidth and y1 >= imgheight:
                x1 = imgwidth - 1
                y1 = imgheight - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                #Save each patch into file directory
                cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            elif y1 >= imgheight: # when patch height exceeds the image height
                y1 = imgheight - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                #Save each patch into file directory
                cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            elif x1 >= imgwidth: # when patch width exceeds the image width
                x1 = imgwidth - 1
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                #Save each patch into file directory
                cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
            else:
                #Crop into patches of size MxN
                tiles = image_copy[y:y+M, x:x+N]
                #Save each patch into file directory
                cv2.imwrite('saved_patches/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

    #Save full image into file directory
    cv2.imshow("Patched Image",img)
    # cv2.imwrite("patched.jpg",img)
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    output = "cropping tutorial finished"
    return output

def resizing():
    # link to tutorial https://learnopencv.com/image-resizing-with-opencv/

    # Read the image using imread function
    print("reading in image")
    image = cv2.imread('/Users/akhilkodumuri/Desktop/image.img')
    cv2.imshow('Original Image', image)
    cv2.waitKey(1)
    
    # let's downscale the image using new  width and height
    print("downsizing image")
    down_width = 300
    down_height = 200
    down_points = (down_width, down_height)
    resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

    # let's upscale the image using new  width and height
    print("upscaling image")
    up_width = 600
    up_height = 400
    up_points = (up_width, up_height)
    resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)

    # Display images
    print("scaling images")
    cv2.imshow('Resized Down by defining height and width', resized_down)
    cv2.waitKey(1)
    cv2.imshow('Resized Up image by defining height and width', resized_up)
    cv2.waitKey(1)

    # press any key to close the windows
    cv2.destroyAllWindows()

    # interpolation techniques
    '''
    INTER_AREA: INTER_AREA uses pixel area relation for resampling. This is best suited for reducing the size of an image (shrinking). When used for zooming into the image, it uses the INTER_NEAREST method.
    INTER_CUBIC: This uses bicubic interpolation for resizing the image. While resizing and interpolating new pixels, this method acts on the 4×4 neighboring pixels of the image. It then takes the weights average of the 16 pixels to create the new interpolated pixel.
    INTER_LINEAR: This method is somewhat similar to the INTER_CUBIC interpolation. But unlike INTER_CUBIC, this uses 2×2 neighboring pixels to get the weighted average for the interpolated pixel.
    INTER_NEAREST: The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.    
    '''
    # Scaling Up the image 1.2 times by specifying both scaling factors
    scale_up_x = 1.2
    scale_up_y = 1.2
    # Scaling Down the image 0.6 times specifying a single scale factor.
    scale_down = 0.6

    scaled_f_down = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
    scaled_f_up = cv2.resize(image, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_LINEAR)
    
    # Display images and press any key to check next image
    cv2.imshow('Resized Down by defining scaling factor', scaled_f_down)
    cv2.waitKey(1)
    cv2.imshow('Resized Up image by defining scaling factor', scaled_f_up)
    cv2.waitKey(1)

    # Scaling Down the image 0.6 times using different Interpolation Method
    print("different types of interpolation")
    res_inter_nearest = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_NEAREST)
    res_inter_linear = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
    res_inter_area = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_AREA)

    # Concatenate images in horizontal axis for comparison
    vertical= np.concatenate((res_inter_nearest, res_inter_linear, res_inter_area), axis = 0)
    # Display the image Press any key to continue
    cv2.imshow('Inter Nearest :: Inter Linear :: Inter Area', vertical)
    cv2.waitKey(1)

    cv2.destroyAllWindows()
    return "rezize tutorial finished"

def readAndWrite():
    # link to tutorial https://learnopencv.com/reading-and-writing-videos-using-opencv/

    # Create a video capture object, in this case we are reading the video from a file
    vid_capture = cv2.VideoCapture('/Users/akhilkodumuri/Desktop/Car.mp4')

    # Check if the video file is corrupted or not
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        # Get frame rate information

        fps = int(vid_capture.get(5))
        print("Frame Rate : ",fps,"frames per second")	

        # Get frame count
        frame_count = vid_capture.get(7)
        print("Frame count : ", frame_count)
    
    # this loop goes through every frame in the video
    print("about to loop thorough every frame in the video")
    while(vid_capture.isOpened()):
        # vCapture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            k = cv2.waitKey(1)
            # 113 is ASCII code for q key
            if k == 113:
                break
        else:
            break


    # Release the objects
    vid_capture.release()
    cv2.destroyAllWindows()

    
    return "reading and writing tutorial finished"

def imageRotationAndTranslation():
    # link to tutorial https://learnopencv.com/image-rotation-and-translation-using-opencv/

    print("starting image rotation example")

    # Reading the image
    image = cv2.imread('image.jpg')

    # dividing height and width by 2 to get the center of the image
    height, width = image.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)

    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

    cv2.imshow('Original image', image)
    cv2.imshow('Rotated image', rotated_image)
    # wait indefinitely, press any key on keyboard to exit
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # save the rotated image to disk
    # cv2.imwrite('rotated_image.jpg', rotated_image)

    print("starting image translation example")

    # read the image 
    image = cv2.imread('image.jpg')
    # get the width and height of the image
    height, width = image.shape[:2]
    # get tx and ty values for translation
    # you can specify any value of your choice
    tx, ty = width / 4, height / 4

    # create the translation matrix using tx and ty, it is a NumPy array 
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)

    # apply the translation to the image
    translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))
    # display the original and the Translated images
    cv2.imshow('Translated image', translated_image)
    cv2.imshow('Original image', image)
    cv2.waitKey(0)
    # save the translated image to disk
    cv2.imwrite('translated_image.jpg', translated_image)
    return "finished rotation and translation tutorial"

def annotatingImages():
    print("start annotating Images tutorial")
    # Read Images
    img = cv2.imread('sample.jpg')
    # Display Image
    cv2.imshow('Original Image',img)
    cv2.waitKey(1)
    # Print error message if image is null
    if img is None:
        print('Could not read image')
    # Draw line on image
    imageLine = img.copy()
    # Draw the image from point A to B
    pointA = (200,80)
    pointB = (450,80)
    cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow('Image Line', imageLine)
    cv2.waitKey(1)

    print("creating a circle annotation")
    # Make a copy of image
    imageCircle = img.copy()
    # define the center of circle
    circle_center = (415,190)
    # define the radius of the circle
    radius =100
    #  Draw a circle using the circle() Function
    # thickness=-1 fills the circle
    cv2.circle(imageCircle, circle_center, radius, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA) 
    # Display the result
    cv2.imshow("Image Circle",imageCircle)
    cv2.waitKey(1)

    # make a copy of the original image
    imageRectangle = img.copy()
    # define the starting and end points of the rectangle
    start_point =(300,115)
    end_point =(475,225)
    # draw the rectangle
    cv2.rectangle(imageRectangle, start_point, end_point, (0, 0, 255), thickness= 3, lineType=cv2.LINE_8) 
    # display the output
    cv2.imshow('imageRectangle', imageRectangle)
    cv2.waitKey(1)
    print("adding text to the picture")

    # make a copy of the original image
    imageText = img.copy()
    #let's write the text you want to put on the image
    text = 'I am a Happy dog!'
    #org: Where you want to put the text
    org = (50,350)
    # write the text on the input image
    cv2.putText(imageText, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))
    # display the output image with text over it
    cv2.imshow("Image Text",imageText)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return "finished annotating image tutorial"

def colorSpaces():
    # link to tutorial https://learnopencv.com/color-spaces-in-opencv-cpp-python/
    # Open cv loads images in the BGR Color Space
    bright = cv2.imread('cube1.jpg')
    dark = cv2.imread('cube8.jpg')

    # LAB Color Space
    brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

    # bRCrCb Color Space
    brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

    # HSV Color Space
    brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
    darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)



    return "Finished color spaces tutorial"

def convolution():
    image = cv2.imread('test.jpg')
    """
    Apply identity kernel
    """
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    # filter2D() function can be used to apply kernel to an image.
    # Where ddepth is the desired depth of final image. ddepth is -1 if...
    # ... depth is same as original or source image.
    identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

    # We should get the same image
    cv2.imshow('Original', image)
    cv2.imshow('Identity', identity)

    cv2.waitKey()
    cv2.imwrite('identity.jpg', identity)
    cv2.destroyAllWindows()
    
    # end identity kernel

    # Begin custom blur of image using 2d convolution kernel
    """
    Apply blurring kernel
    """
    kernel2 = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

    cv2.imshow('Original', image)
    cv2.imshow('Kernel Blur', img)

    cv2.waitKey()
    cv2.imwrite('blur_kernel.jpg', img)
    cv2.destroyAllWindows()
    # End custom blur of image using 2d convolution kernel

    # Begin blur using Opencv blur function 
    """
    Apply blur using `blur()` function
    """
    img_blur = cv2.blur(src=image, ksize=(5,5)) # Using the blur function to blur an image where ksize is the kernel size

    # Display using cv2.imshow()
    cv2.imshow('Original', image)
    cv2.imshow('Blurred', img_blur)

    cv2.waitKey()
    cv2.imwrite('blur.jpg', img_blur)
    cv2.destroyAllWindows()
    # End blur using Opencv blur function 

    # Begin guassian blur
    """
    Apply Gaussian blur
    """
    # sigmaX is Gaussian Kernel standard deviation 
    # ksize is kernel size
    gaussian_blur = cv2.GaussianBlur(src=image, ksize=(5,5), \\
    sigmaX=0, sigmaY=0)

    cv2.imshow('Original', image)
    cv2.imshow('Gaussian Blurred', gaussian_blur)
        
    cv2.waitKey()
    cv2.imwrite('gaussian_blur.jpg', gaussian_blur)
    cv2.destroyAllWindows()
    # End guassian blur

    # Begin median blur
    """
    Apply Median blur
    """
    # medianBlur() is used to apply Median blur to image
    # ksize is the kernel size
    median = cv2.medianBlur(src=image, ksize=5)

    cv2.imshow('Original', image)
    cv2.imshow('Median Blurred', median)
        
    cv2.waitKey()
    cv2.imwrite('median_blur.jpg', median)
    cv2.destroyAllWindows()
    # End median blur

    # Begin Sharpening an Image Using Custom 2D-Convolution Kernels
    """
    Apply sharpening using kernel
    """
    kernel3 = np.array([[0, -1,  0],
                    [-1,  5, -1],
                        [0, -1,  0]])
    sharp_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)

    cv2.imshow('Original', image)
    cv2.imshow('Sharpened', sharp_img)
        
    cv2.waitKey()
    cv2.imwrite('sharp_image.jpg', sharp_img)
    cv2.destroyAllWindows()
    # End Sharpening an Image Using Custom 2D-Convolution Kernels

    # Begin Applying Bilateral Filtering to an Image in OpenCV
    """
    Apply Bilateral Filtering
    """
    # Using the function bilateralFilter() where d is diameter of each...
    # ...pixel neighborhood that is used during filtering.
    # sigmaColor is used to filter sigma in the color space.
    # sigmaSpace is used to filter sigma in the coordinate space.
    bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)

    cv2.imshow('Original', image)
    cv2.imshow('Bilateral Filtering', bilateral_filter)

    cv2.waitKey(0)
    cv2.imwrite('bilateral_filtering.jpg', bilateral_filter)
    cv2.destroyAllWindows()
    # End Applying Bilateral Filtering to an Image in OpenCV
    return "Finished convolution tutorial"

def imageThresholding():
    # Read image
    src = cv2.imread("threshold.png", cv2.IMREAD_GRAYSCALE)

    # Set threshold and maxValue
    thresh = 0
    maxValue = 255 

    print("Starting binray thresholding")
    # Basic threshold example using binary thresholding
    th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)

    print("Starting inverse binary thresholding")
    th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV)


    print("Starting truncate thresholding")
    """
    # Truncate Threshold
    if src(x,y) > thresh
        dst(x,y) = thresh
    else
        dst(x,y) = src(x,y)
    """
    th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TRUNC)

    return "Image Thresholding Tutorial"

    

    return "Finished thresholding tutorial"

def blobDetection():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    return "Finished blob detection tutorial"

def edgeDetection():
    # Read the original image
    img = cv2.imread('test.jpg') 
    # Display original image
    cv2.imshow('Original', img)
    cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return "Finished edge detection tutorial"

def mouseGUI():
    # Lists to store the points
    top_left_corner=[]
    bottom_right_corner=[]
    def drawRectangle(action, x, y, flags, *userdata):
        # Referencing global variables 
        global top_left_corner, bottom_right_corner
        # Mark the top left corner, when left mouse button is pressed
        if action == cv2.EVENT_LBUTTONDOWN:
            top_left_corner = [(x,y)]
            # When left mouse button is released, mark bottom right corner
        elif action == cv2.EVENT_LBUTTONUP:
            bottom_right_corner = [(x,y)]    
            # Draw the rectangle
            cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2,8)
            cv2.imshow("Window",image)
    # Read Images
    image = cv2.imread("../Input/sample.jpg")
    # Make a temporary image, will be useful to clear the drawing
    temp = image.copy()
    # Create a named window.
    cv2.namedWindow("Window")
    # highgui function called when mouse events occur
    cv2.setMouseCallback("Window", drawRectangle)
    k=0
    while k!=113:
    cv2.imshow("Window", image)
    k = cv2.waitKey(0)
    if (k == 99):
        image= temp.copy()
        cv2.imshow("Window", image)
    cv2.destroyAllWindows()

    
    return "Finished mouse GUI tutorial"

def trackbar():
    maxScaleUp = 100
    scaleFactor = 1
    windowName = "Resize Image"
    trackbarValue = "Scale"

    # read the image
    image = cv2.imread("../Input/sample.jpg")

    # Create a window to display results and  set the flag to Autosize
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

    # Callback functions
    def scaleImage(*args):
        # Get the scale factor from the trackbar 
        scaleFactor = 1+ args[0]/100.0
        # Resize the image
        scaledImage = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
        cv2.imshow(windowName, scaledImage)

    # Create trackbar and associate a callback function
    cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)

    # Display the image
    cv2.imshow(windowName, image)
    c = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "Finished trackbar tutorial"

if __name__ == "__main__":
    mouseGUI()