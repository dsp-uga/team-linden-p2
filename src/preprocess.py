import cv2
import os
import glob
from vidstab import VidStab


#convert the image frames to videos

#video_name = '/home/afarahani/Projects/video.mp4'
def image_to_video(image_path,out_video_path):
    for image_folder in image_path:
        head, tail = os.path.split(image_folder)
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join(out_video_path, tail+'.mp4'), 0, 50, (width,height))

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

#conveting the video to image frames and applying auto correction for illumination and smoothing
def processing_frame_from_video(input_path, output_path):
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
            print(image.shape)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#auto illumination correction
            image = clahe.apply(image)        

            image = cv2.bilateralFilter(image,7,30,30) #smoothing the image
            cv2.imwrite(image_path+'/frame'+str(count).zfill(4)+'.png', image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

def main():
    image_path = glob.glob('/home/afarahani/Projects/project2/dataset/data/data/train_org/data/*')
    out_video_path = '/home/afarahani/Projects/output/'
    image_to_video(image_path,out_video_path)

    input_video_filepath = glob.glob('/home/afarahani/Projects/output/*')
    output_video_filepath = '/home/afarahani/Projects/stab_output/'
    video_stabilizer(input_video_filepath, output_video_filepath)

    input_path = glob.glob('/home/afarahani/Projects/stab_output/*')
    output_path = '/home/afarahani/Projects/stab_images'
    processing_frame_from_video(input_path, output_path)




if __name__ == "__main__":
    main()