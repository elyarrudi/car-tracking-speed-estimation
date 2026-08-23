from pathlib import Path
import cv2
PROJECT_DIR=Path(__file__).resolve().parent
video_path=PROJECT_DIR/"bb.mp4"
cap=cv2.VideoCapture(str(video_path))
ret,frame=cap.read()
if not ret:
    print(f"Error: Could not open video: {video_path}")
    exit()
frame0=frame.copy()
points=[]
no=[]
paused=True
scale=1
first_line_y_top=None
first_line_y_bottom=None
def click_event(event,x,y,flags,param):
    global frame,frame0,points,no,first_line_y_top,first_line_y_bottom
    if event==cv2.EVENT_LBUTTONDOWN:
        x_original=int(x/scale)
        y_original=int(y/scale)
        points.append((x_original,y_original))
        if len(points)%2==0:
            if len(no)==0:
                x1,y1=points[-2]
                x2,y2=points[-1]
                y_top=min(y1,y2)
                y_bottom=max(y1,y2)
                first_line_y_top=y_top
                first_line_y_bottom=y_bottom
                x_line=x2
                no.append((x_line,y_top,x_line,y_bottom))
                print(f"Line 1: ({x_line},{y_top})->({x_line},{y_bottom})")
            elif len(no)==1:
                x1,y1=points[-2]
                x2,y2=points[-1]
                x_line=x2
                no.append((x_line,first_line_y_top,x_line,first_line_y_bottom))
                print(f"Line 2: ({x_line},{first_line_y_top})->({x_line},{first_line_y_bottom})")
cv2.namedWindow("video")
cv2.setMouseCallback("video",click_event)
while True:
    if not paused:
        ret,frame=cap.read()
        if not ret:
            break
        frame0=frame.copy()
    frame=frame0.copy()
    for line in no:
        x1,y1,x2,y2=line
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    frame_small=cv2.resize(frame,(0,0),fx=scale,fy=scale)
    cv2.imshow("video",frame_small)
    key=cv2.waitKeyEx(30)&0xFF
    if key==32:
        paused=not paused
    elif key==ord("r"):
        points=[]
        no=[]
        first_line_y_top=None
        first_line_y_bottom=None
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        ret,frame=cap.read()
        if ret:
            frame0=frame.copy()
    elif key==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()