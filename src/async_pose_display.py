"""独立显示进程：单帧共享缓冲，生产者try-lock，绝不等待GUI。"""
import multiprocessing as mp
import time


def _display(buffer, shape, lock, version, command, stopping, fps_value, window):
    import cv2
    import numpy as np
    from collections import deque
    cv2.setNumThreads(1)
    shared=np.frombuffer(buffer,dtype=np.uint8).reshape(shape)
    cv2.namedWindow(window,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window,1280,round(1280*shape[0]/shape[1]))
    seen=0;recent=deque();image=None
    try:
        while not stopping.value:
            # 锁仅用于复制像素，imshow/waitKey期间不持锁。
            if lock.acquire(False):
                try:
                    if version.value!=seen:
                        image=shared.copy();seen=version.value
                finally:lock.release()
            if image is not None:
                now=time.monotonic();recent.append(now)
                while recent and recent[0]<now-2:recent.popleft()
                fps_value.value=(len(recent)-1)/max(now-recent[0],.001)
                cv2.putText(image,f"Display FPS: {fps_value.value:.1f}",(10,82),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4)
                cv2.putText(image,f"Display FPS: {fps_value.value:.1f}",(10,82),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.imshow(window,image);image=None
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):command.value=ord('q');break
            if key==ord('r'):command.value=key
            time.sleep(.001)
    finally:cv2.destroyAllWindows()


class AsyncPoseDisplay:
    def __init__(self,shape,window):
        import numpy as np
        ctx=mp.get_context('spawn');self.shape=tuple(shape)
        self.buffer=ctx.RawArray('B',int(np.prod(shape)))
        self.lock=ctx.Lock();self.version=ctx.RawValue('L',0)
        self.command=ctx.RawValue('i',-1);self.stopping=ctx.RawValue('b',0);self.fps=ctx.RawValue('d',0)
        self.process=ctx.Process(target=_display,args=(self.buffer,self.shape,self.lock,self.version,self.command,self.stopping,self.fps,window),daemon=True)
        self.process.start();self.dropped=0

    def submit(self,image):
        import numpy as np
        if tuple(image.shape)!=self.shape:raise ValueError('Display image dimensions changed')
        if not self.process.is_alive():return False
        if not self.lock.acquire(False):self.dropped+=1;return False
        try:
            np.copyto(np.frombuffer(self.buffer,dtype=np.uint8).reshape(self.shape),image)
            self.version.value+=1
        finally:self.lock.release()
        return True

    def poll_key(self):
        # 只有Q/R控制消息，不排队图像。
        key=self.command.value
        if key!=ord('q'):self.command.value=-1
        return key

    def close(self):
        self.stopping.value=1;self.process.join(timeout=1)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
