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



if __name__ == "__main__":
    cropping()