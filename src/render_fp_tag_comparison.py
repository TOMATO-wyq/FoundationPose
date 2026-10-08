#!/usr/bin/env python3
"""同步绘制保存的 FP / Tag 位姿；按源帧配对，不重新运行推理。"""
import csv
import itertools
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/home/tomato/6Dpose/outputs')
OUT = ROOT / 'fp_tag_evaluation_075_final'
CORNERS = np.array(list(itertools.product((-.175, .175), repeat=3)))
EDGES = [(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(CORNERS[i] != CORNERS[j]) == 1]

def draw_pose(image, pose, k, color, thickness):
    # 箱体边长35cm，模型原点位于中心；所有投影均使用同一帧内参。
    points = np.vstack((CORNERS, [0,0,0], [.12,0,0], [0,.12,0], [0,0,.12]))
    camera = points @ pose[:3,:3].T + pose[:3,3]
    if np.any(camera[:,2] <= 0):
        return
    projected = camera @ k.T
    pixels = np.rint(projected[:,:2] / projected[:,2:3]).astype(int)
    pixels = np.clip(pixels, -10000, 10000)
    for i,j in EDGES:
        cv2.line(image, tuple(pixels[i]), tuple(pixels[j]), color, thickness, cv2.LINE_AA)
    # 坐标轴颜色统一：X红、Y绿、Z蓝。Tag模型轴与原OBJ轴尚未独立标定。
    for i,c,label in [(9,(0,0,255),'X'),(10,(0,255,0),'Y'),(11,(255,0,0),'Z')]:
        cv2.arrowedLine(image,tuple(pixels[8]),tuple(pixels[i]),c,thickness,cv2.LINE_AA,tipLength=.15)
        cv2.putText(image,label,tuple(pixels[i]),cv2.FONT_HERSHEY_SIMPLEX,.6,c,2,cv2.LINE_AA)

for kind in ('static','dynamic'):
    sequence = ROOT / f'd435i_{kind}_sequence'
    with (OUT / kind / 'per_frame.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    k=np.loadtxt(sequence/'cam_K.txt')
    first=cv2.imread(str(sequence/'rgb'/rows[0]['rgb_file']))
    h,w=first.shape[:2]
    # 左FP、右Tag、下方叠加：避免两种框覆盖后无法区分。
    size=(w*2,h*2)
    writer=cv2.VideoWriter(str(OUT/kind/'comparison_labeled.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
    if not writer.isOpened(): raise RuntimeError('MP4 encoder unavailable')
    previous_time=None
    for index,row in enumerate(rows):
        rgb=cv2.imread(str(sequence/'rgb'/row['rgb_file']))
        if rgb is None: raise RuntimeError(row['rgb_file'])
        fp=rgb.copy(); tag=rgb.copy(); overlay=rgb.copy()
        name=Path(row['rgb_file']).stem+'.txt'
        fp_path=ROOT/f'd435i_{kind}_fp'/'ob_in_cam'/name
        tag_path=ROOT/f'd435i_{kind}_tag_075'/'tag_in_cam'/name
        if fp_path.exists():
            pose=np.loadtxt(fp_path); draw_pose(fp,pose,k,(0,255,0),3); draw_pose(overlay,pose,k,(0,255,0),3)
        if tag_path.exists():
            pose=np.loadtxt(tag_path)
            # 标签面中心 -> 箱体中心：沿标签负Z移动175mm，无任何结果拟合。
            pose[:3,3] -= .175*pose[:3,2]
            draw_pose(tag,pose,k,(255,255,0),3); draw_pose(overlay,pose,k,(255,255,0),2)
        canvas=np.zeros((h*2,w*2,3),np.uint8)
        canvas[:h,:w]=fp; canvas[:h,w:]=tag; canvas[h:,:w]=overlay
        texts=[(20,35,'FP: green cube'),(w+20,35,'Tag-derived cube: cyan'),(w-620,h+35,'GREEN: FP / CYAN: Tag reference')]
        error=row['translation_error_mm']; angle=row['rotation_cube_sym_deg']
        info=[f'{kind}: source frame {row["source_frame"]}',f'Source time: {float(row["time_s"]):.2f} s',f'Center difference: {float(error):.1f} mm' if error else 'Center difference: unavailable',f'Cube symmetry angle: {float(angle):.2f} deg' if angle else 'Angle: unavailable','Axes: X red / Y green / Z blue','Tag axes != calibrated OBJ axes','Missing result: no box (not reused)','Saved poses; playback is not inference FPS']
        for n,text in enumerate(info): texts.append((w+25,h+55+n*48,text))
        for x,y,text in texts:
            cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),5,cv2.LINE_AA)
            cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
        # 按源时间差补重复展示帧，保留缺失RGB-D时的时间间隔。
        t=float(row['time_s'])
        if previous_time is not None:
            for _ in range(max(0,round((t-previous_time)*30)-1)): writer.write(cv2.resize(previous_canvas,(w,h)))
        writer.write(cv2.resize(canvas,(w,h))); previous_canvas=canvas; previous_time=t
        if index==0 or (kind=='dynamic' and row['source_frame']=='1829'):
            cv2.imwrite(str(OUT/kind/f'preview_{row["source_frame"]}.jpg'),canvas)
    writer.release()
    cap=cv2.VideoCapture(str(OUT/kind/'comparison_labeled.mp4'))
    ok,im=cap.read()
    assert ok and im.shape[:2]==(h,w)
    print(kind,'video frames',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),flush=True)
    cap.release()
