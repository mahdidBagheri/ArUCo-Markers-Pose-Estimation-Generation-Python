import cv2

size = (640,480)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

for i in range(0,1622):
    if(i%10 == 0):
        print(f"{i/1622:0.2f}")

    frame = cv2.imread(f"frames/frame_{i:2d}.jpg".replace(' ', '0'))
    out.write(frame)

out.release()

