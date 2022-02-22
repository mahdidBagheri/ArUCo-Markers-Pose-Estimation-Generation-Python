import cv2
import glob

size = (640,480)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

number_of_frames = len(glob.glob("frames/*.jpg"))
for i in range(0,number_of_frames):
    if(i%10 == 0):
        print(f"{i/number_of_frames:0.2f}")

    frame = cv2.imread(f"frames/frame_{i:2d}.jpg".replace(' ', '0'))
    out.write(frame)

out.release()

