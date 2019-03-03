import cv2
import os
import argparse
import shutil
from vidstab import VidStab

#video_name = '/home/afarahani/Projects/video.mp4'
def image_to_video(image_path,out_video_path):
    for image_folder in image_path:
        head, tail = os.path.split(image_folder)
        images = [img for img in os.listdir(image_folder) \
                  if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join(out_video_path, tail+'.mp4'), 
                                0, 50, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


#stabilizing the video
def video_stabilizer(input_video_filepath, output_video_filepath):
    # input_video_filepath = glob.glob('/home/afarahani/Projects/output/*')
    # output_video_filepath = '/home/afarahani/Projects/stab_output/'
    for video_path in input_video_filepath:
        head, tail = os.path.split(video_path)
        stabilizer = VidStab()
        # black borders
        stabilizer.stabilize(input_path=video_path, 
                             output_path=output_video_filepath+tail, 
                             border_type='black')

#conveting the video to image frames and applying auto correction 
#for illumination and smoothing
def processing_frame_from_video(input_path, output_path, args):
    # input_path = glob.glob('/home/afarahani/Projects/stab_output/*')
    # output_path = '/home/afarahani/Projects/stab_images'
    for item in input_path:
        head, tail = os.path.split(item)
        vidcap = cv2.VideoCapture(item)
        success,image = vidcap.read()
        image_path = os.path.join(output_path, tail[:-4])
        try:
            if not os.path.exists(image_path):
                os.makedirs(image_path)
        except OSError:
            print ('Error: Creating directory of data')
            
        count = 0
        while success:
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #print(image.shape)
            if args.clahe:
                # auto illumin ation correction
                clahe = cv2.createCLAHE(clipLimit=args.clipLimit, \
                                        tileGridSize=(args.tileGridX, \
                                                      args.tileGridY))
                image = clahe.apply(image)

            if args.bf:
                # smoothing the image
                image = cv2.bilateralFilter( image, args.bf_d, \
                                            args.bf_sc, args.bf_ss ) 

            if args.gb:
                image = cv2.GaussianBlur(image,(args.gb_x,args.gb_y), \
                                         args.gb_border_type)

            cv2.imwrite(image_path+'/frame'+str(count).zfill(4)+'.png', image) 
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1

# Main function
def main(args):
    image_path = os.path.normpath(args.sourceDir)
    output_path = os.path.normpath(args.sourceDir)
    
    # Did the user ask for download data?
    if args.sourceAddr:
        # make sure the target source dir is empty
        if os.path.exists(image_path):
            shutil.rmtree(image_path)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        os.system("gsutil cp -r " + args.sourceAddr + " " + image_path )
        # Now untar all tar files there
        for file in os.listdir(image_path + "/data/"):
            if file.endswith(".tar"):
                os.system('tar xvf ' + file)
    else:
        # Check to see if expected directories exist
        if not os.path.exists(image_path) \
        or not os.path.exists(image_path + "/data" ) \
        or not os.path.exists(image_path + "/masks/"):
            raise Exception("The source directory you specified does not " + \
                            "conform with expectations.")

    # make sure expected output directories exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # make a temp directory for videos
    if not os.path.exists(image_path + "/.video"):
        os.mkdir(image_path + "/.video")
    if not os.path.exists(image_path + "/.video"):
        os.mkdir(image_path + "/.video_stab")
    
    # Make videos from images for later use
    image_to_video(image_path + "/data/*", image_path + "/.video")

    # apply video stabalizer if asked for
    if args.vs:
        video_stabilizer(image_path + "/.video", image_path + "/.video_stab")
        processing_frame_from_video(image_path + "/.video_stab", 
                                    output_path, args)
    else:
        processing_frame_from_video(image_path + "/.video", output_path, args)

    # Remove temp directories
    shutil.rmtree(image_path + "/.video")
    shutil.rmtree(image_path + "/.video_stab")
    

# Parse arguments and call main    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description='This ' + \
            'is part of the UGA CSCI 8360 Project 2 - . Please visit our ' + \
            'GitHub project at https://github.com/dsp-uga/team-linden-p2 ' + \
            'for more information regarding data organization ' + \
            'expectations and examples on how to execute these scripts.')

    # Options for downloading data or specifying source directory
    parser.add_argument('-s', '--sourceDir', required=True,
                       help='Root directory for data that agrees with our '+ \
                       'organization standards.')
    parser.add_argument('-g', '--sourceAddr',
                       help='The address for source data that agrees with ' + \
                       'our organization standards. WARN: This will write ' + \
                       'to the sourceDir, deleting what is already there!!!')
    
    parser.add_argument('-o', '--outputDir', required=True,
                        help='Directory to store all png cilia predictions')

    parser.add_argument('-clahe', '--clahe', action='store_true',
                        help='Signify if you want clahe image processing')
    parser.add_argument('-clipLimit', '--clipLimit', default=2, type=int,
                        help='clahe: clip limit variable')
    parser.add_argument('-tileGridX', '--tileGridX',  default=8, type=int,
                        help='clahe: Tile Grid x-dim size')
    parser.add_argument('-tileGridY', '--tileGridY',  default=8, type=int,
                        help='clahe: Tile Grid y-dim size')

    parser.add_argument('-bf', '--bf', action='store_true',
                        help='Signify if you want Bilateral Filter image ' + \
                        'processing')    
    parser.add_argument('-bf_d', '--bf_d', default=7, type=int,
                        help='Bilateral Filter: The d argument')
    parser.add_argument('-bf_sc', '--bf_sc', default=30, type=float,
                        help='Bilateral Filter: The sc argument')
    parser.add_argument('-bf_ss', '--bf_ss', default=30, type=float,
                        help='Bilateral Filter: The ss argument')
    
    parser.add_argument('-gb', '--gb', action='store_true',
                        help='Signify if you want Gaussian Blur image ' + \
                        'processing')
    parser.add_argument('-gb_x', '--gb_x', default=5, type=float,
                        help='Gaussian Blur: sigma x argument')
    parser.add_argument('-gb_y', '--gb_y', default=5, type=float,
                        help='Gaussian Blur: sigma y argument')
    parser.add_argument('-gb_border_type', '--gb_border_type', 
                        default=0, type=int,
                        help='Gaussian Blur: border type argument')

    parser.add_argument('-vs','--vs',action='store_true',
                       help='Signify if you would like vidoe stabalizing ' + \
                       'applied to a movie generated by frames')

    args = parser.parse_args()
    
    main(args)


