"""独立CPU进程，队列最多一个待处理帧，主RGB-D流程从不等待结果。"""
import multiprocessing as mp
import queue
import time


def worker(inbox, outbox, size, tag_id, distortion):
    import ctypes,sys
    from pathlib import Path
    import cv2,numpy as np
    root=Path.home()/'apriltag_ws/install/apriltag/lib'
    ctypes.CDLL(str(root/'libapriltag.so.3'), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(root/'python3.10/site-packages'))
    import apriltag
    detector=apriltag.apriltag('tag36h11',threads=1,decimate=1.0)
    cv2.setNumThreads(1)
    points=np.array([[-size/2,-size/2,0],[size/2,-size/2,0],[size/2,size/2,0],[-size/2,size/2,0]],np.float64)
    while True:
        item=inbox.get()
        if item is None:return
        gray,k,stamp,sequence=item
        result=dict(stamp=stamp,sequence=sequence,pose=None)
        detections=detector.detect(gray)
        for d in detections:
            if d['id']!=tag_id:continue
            corners=np.asarray(d['lb-rb-rt-lt'],np.float64)
            ok,rvec,tvec=cv2.solvePnP(points,corners,k,np.asarray(distortion,np.float64),flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rvec)[0];pose[:3,3]=tvec.ravel()
                result['pose']=pose.tolist()
                break
        try:outbox.get_nowait()
        except queue.Empty:pass
        try:outbox.put_nowait(result)
        except queue.Full:pass


class LiveTagWorker:
    def __init__(self,size,tag_id,hz,distortion):
        ctx=mp.get_context('spawn')
        self.inbox=ctx.Queue(maxsize=1);self.outbox=ctx.Queue(maxsize=1)
        self.process=ctx.Process(target=worker,args=(self.inbox,self.outbox,size,tag_id,distortion),daemon=True)
        self.process.start();self.period=1/hz;self.last=0;self.result=None

    def submit(self,rgb,k,stamp,sequence):
        import cv2
        now=time.monotonic()
        if now-self.last<self.period:return
        self.last=now
        # 只排队灰度图；满队列直接丢帧，不让相机回调等待Tag。
        try:self.inbox.put_nowait((cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),k.copy(),stamp,sequence))
        except queue.Full:pass

    def poll(self):
        try:
            while True:self.result=self.outbox.get_nowait()
        except queue.Empty:return self.result

    def close(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=1)
