#!/usr/bin/env python3
"""D435i 实时 YOLO 分割测试，订阅 ROS 彩色图，保持原图尺寸的目标 mask。"""
import argparse,datetime,json,os,threading,time
from pathlib import Path
ROOT=Path('/home/tomato/code/box_seg_annotation')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'config/matplotlib'))
import cv2,numpy as np
# 预览使用 OpenCV 自带的 Qt 插件，不加载 Labelme 的 PyQt5。

def decode_image(msg):
 """直接解码 RGB8/BGR8，支持每行 padding，不依赖 cv_bridge。"""
 if msg.encoding.lower() not in ('rgb8','bgr8'):raise ValueError(f'不支持 {msg.encoding}')
 if msg.step<msg.width*3 or len(msg.data)<msg.height*msg.step:raise ValueError('图像字节长度无效')
 frame=np.frombuffer(msg.data,dtype=np.uint8,count=msg.height*msg.step).reshape(msg.height,msg.step)[:,:msg.width*3].reshape(msg.height,msg.width,3)
 if msg.encoding.lower()=='rgb8':frame=frame[:,:,::-1]
 return np.ascontiguousarray(frame).copy()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--weights',type=Path,default=ROOT/'runs/box_seg_trial_20261004/weights/best.pt')
 p.add_argument('--topic',default='/camera/camera/color/image_raw');p.add_argument('--conf',type=float,default=.5)
 p.add_argument('--imgsz',type=int,default=640);p.add_argument('--device',default='0')
 p.add_argument('--no-display',action='store_true');p.add_argument('--max-frames',type=int,default=0)
 p.add_argument('--save-first',action='store_true',help='保存首帧 RGB/mask/叠加图，用于测试')
 p.add_argument('--timeout',type=float,default=15);a=p.parse_args()
 if not a.weights.is_file():p.error('权重文件不存在')
 if not 0<a.conf<1 or a.max_frames<0:p.error('conf 或 max-frames 无效')
 import torch,rclpy
 from rclpy.node import Node
 from rclpy.qos import qos_profile_sensor_data
 from sensor_msgs.msg import Image
 from ultralytics import YOLO
 if a.device!='cpu' and not torch.cuda.is_available():raise SystemExit('未发现 CUDA，检查环境或用 --device cpu')
 model=YOLO(str(a.weights))
 if model.task!='segment' or model.names.get(0) not in ('blue_kfs','purple_box'):raise SystemExit('需要类别 0=blue_kfs 的实例分割权重')
 # 兼容历史单类别权重，预测显示统一使用新名称。
 model.model.names[0]='blue_kfs'
 # 预热后再启动图像订阅，避免相机启动阶段积压旧帧。
 model.predict(np.zeros((720,1280,3),np.uint8),imgsz=a.imgsz,device=a.device,conf=a.conf,verbose=False)
 rclpy.init();node=Node('box_yolo_d435i_test');lock=threading.Lock();latest=[None,None,0.,0];sequence=0
 def on_image(msg):
  nonlocal sequence
  try:
   frame=decode_image(msg);stamp=dict(sec=msg.header.stamp.sec,nanosec=msg.header.stamp.nanosec,frame_id=msg.header.frame_id)
   with lock:sequence+=1;latest[:]=[frame,stamp,time.monotonic(),sequence]
  except Exception as e:node.get_logger().error(str(e),throttle_duration_sec=5)
 sub=node.create_subscription(Image,a.topic,on_image,qos_profile_sensor_data)
 thread=threading.Thread(target=rclpy.spin,args=(node,),daemon=True);thread.start()
 title='D435i YOLO SEG | left: prediction / right: binary mask | SPACE save | Q quit'
 out=None;processed=0;last_seq=-1;last_received=time.monotonic();reports=[];last_report=0.
 def save(rgb,mask,overlay,stamp,confidences):
  nonlocal out
  if out is None:
   out=ROOT/'live_tests'/datetime.datetime.now().strftime('d435i_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
  key=datetime.datetime.now().strftime('%H%M%S_%f')
  for suffix,data in [('rgb',rgb),('mask',mask),('overlay',overlay)]:
   if not cv2.imwrite(str(out/f'{key}_{suffix}.png'),data):raise RuntimeError('保存图片失败')
  (out/f'{key}.json').write_text(json.dumps(dict(weights=str(a.weights.resolve()),stamp=stamp,threshold=a.conf,confidence=confidences,width=rgb.shape[1],height=rgb.shape[0],selection='highest_confidence_blue_kfs',detected=bool(confidences)),indent=2))
  print('已保存 RGB、0/255 mask、叠加图：',out/key,flush=True)
 try:
  if not a.no_display:cv2.namedWindow(title,cv2.WINDOW_NORMAL);cv2.resizeWindow(title,1280,400)
  print('等待 RGB：',a.topic,'模型：',a.weights,flush=True)
  while rclpy.ok():
   with lock:frame,stamp,received,seq=latest
   now=time.monotonic()
   if frame is None or seq==last_seq or now-received>2:
    if now-last_received>a.timeout:raise RuntimeError('RGB 已超过等待超时，检查相机连接/话题')
    if not a.no_display:
     if frame is None or now-received>2:
      waiting=np.zeros((360,1280,3),np.uint8);cv2.putText(waiting,'Waiting for fresh D435i RGB ... Q: quit',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),2);cv2.imshow(title,waiting)
     if cv2.waitKey(10)&255 in (ord('q'),27):break
    else:time.sleep(.01)
    continue
   last_seq=seq;last_received=now;start=time.perf_counter()
   r=model.predict(frame,imgsz=a.imgsz,device=a.device,conf=a.conf,classes=[0],retina_masks=True,verbose=False)[0]
   ms=(time.perf_counter()-start)*1000;overlay=r.plot();mask=np.zeros(frame.shape[:2],np.uint8)
   confidences=r.boxes.conf.cpu().tolist()
   if r.masks is not None and confidences:
    index=int(np.argmax(confidences));mask=(r.masks.data[index].cpu().numpy()>.5).astype(np.uint8)*255
    if mask.shape!=frame.shape[:2]:mask=cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
   processed+=1
   reports.append(dict(frame=processed,stamp=stamp,detected=bool(confidences),confidence=confidences,inference_ms=round(ms,2),mask_pixels=int(np.count_nonzero(mask))))
   if now-last_report>=2:
    print(f'frame={processed} detections={len(confidences)} conf={max(confidences,default=0):.3f} inference={ms:.1f}ms',flush=True);last_report=now
   if a.save_first and processed==1:save(frame,mask,overlay,stamp,confidences)
   if not a.no_display:
    display=np.hstack([overlay,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)])
    cv2.rectangle(display,(0,0),(display.shape[1],45),(0,0,0),-1)
    cv2.putText(display,f'conf threshold={a.conf:.2f} | inference={ms:.1f}ms | target={"FOUND" if confidences else "NONE"} | SPACE save, Q quit',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,0),2)
    cv2.imshow(title,display);key=cv2.waitKey(1)&255
    if key in (ord('q'),27):break
    if key==32:save(frame,mask,overlay,stamp,confidences)
    if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
   if a.max_frames and processed>=a.max_frames:break
 except KeyboardInterrupt:pass
 finally:
  rclpy.shutdown();thread.join(timeout=3);node.destroy_node()
  if not a.no_display:cv2.destroyAllWindows()
  if out:
   summary=dict(weights=str(a.weights),frames=processed,detected_frames=sum(x['detected'] for x in reports),mean_inference_ms=float(np.mean([x['inference_ms'] for x in reports])) if reports else None,reports=reports)
   (out/'test_report.json').write_text(json.dumps(summary,indent=2));print('测试记录：',out/'test_report.json',flush=True)
if __name__=='__main__':main()
