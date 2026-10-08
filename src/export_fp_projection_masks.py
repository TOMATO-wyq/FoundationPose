#!/usr/bin/env python3
"""仅使用 FP 位姿投影 OBJ 生成 mask；不读取实测深度、不做遮挡筛选。"""
import argparse,json,sys,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/home/tomato/code/yolo_official/scripts')
from export_fp_trial import render

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--sequence',type=Path,default=Path('/home/tomato/6Dpose/outputs/d435i_static_sequence'))
 p.add_argument('--poses',type=Path,default=Path('/home/tomato/6Dpose/outputs/d435i_static_fp/ob_in_cam'))
 p.add_argument('--mesh',type=Path,default=Path('/home/tomato/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj'))
 p.add_argument('--start-sample',type=int,default=1,help='从抽样后的第几张开始，按 1 计数');p.add_argument('--stride',type=int,default=30);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.stride<1 or a.start_sample<1:p.error('stride/start-sample 必须大于零')
 if a.output.exists():p.error('输出目录已存在，禁止覆盖')
 vs=[];faces=[]
 for line in a.mesh.read_text().splitlines():
  if line.startswith('v '):vs.append(list(map(float,line.split()[1:4])))
  elif line.startswith('f '):faces.append([int(v.split('/')[0])-1 for v in line.split()[1:]])
 vs=np.asarray(vs);k=np.loadtxt(a.sequence/'cam_K.txt');out=a.output
 for name in ['images','labelme','masks','masks_projected','labels','preview']:(out/name).mkdir(parents=True)
 records=[]
 for src in sorted((a.sequence/'rgb').glob('*.png'))[::a.stride][a.start_sample-1:]:
  posefile=a.poses/(src.stem+'.txt')
  if not posefile.exists():records.append(dict(frame=src.stem,status='missing_pose'));continue
  im=cv2.imread(str(src));h,w=im.shape[:2];pose=np.loadtxt(posefile)
  # 唯一 mask 来源是模型投影，完全跳过 measured_depth/30mm 判断。
  full=np.isfinite(render(vs,faces,pose,k,h,w));raw=full.astype(np.uint8)*255
  contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  polys=[cv2.approxPolyDP(c,1.5,True).reshape(-1,2) for c in contours if cv2.contourArea(c)>=25]
  polys=[xy for xy in polys if len(xy)>=3]
  if len(polys)!=1:records.append(dict(frame=src.stem,status='needs_review',components=len(polys)));continue
  poly=polys[0];mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[poly],255)
  shutil.copy2(src,out/'images'/src.name)
  cv2.imwrite(str(out/'masks_projected'/src.name),raw);cv2.imwrite(str(out/'masks'/src.name),mask)
  data=dict(version='6.3.1',flags={'reviewed':False},shapes=[dict(label='blue_kfs',points=poly.tolist(),group_id=1,shape_type='polygon',flags={})],imagePath='../images/'+src.name,imageData=None,imageHeight=h,imageWidth=w)
  (out/'labelme'/f'{src.stem}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
  label=out/'labels'/f'{src.stem}.txt';label.write_text('0 '+' '.join(f'{v:.8f}' for v in (poly/[w,h]).ravel())+'\n')
  from ultralytics.data.utils import verify_image_label
  check=verify_image_label((str(out/'images'/src.name),str(label),'',False,1,0,0,False))
  if check[0] is None or len(check[3])!=1:raise RuntimeError(str(check[-1]))
  overlay=im.copy();overlay[mask>0]=(overlay[mask>0]*.65+[0,89.25,0]).astype(np.uint8);cv2.polylines(overlay,[poly],True,(0,255,0),2)
  cv2.putText(overlay,'FP projection ONLY '+src.stem,(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2);cv2.imwrite(str(out/'preview'/f'{src.stem}.jpg'),overlay)
  records.append(dict(frame=src.stem,status='ok',vertices=len(poly),pixels=int((mask>0).sum()),polygon_projection_iou=float(np.logical_and(full,mask>0).sum()/np.logical_or(full,mask>0).sum())))
 ok=[x for x in records if x['status']=='ok'];selected=np.linspace(0,len(ok)-1,min(12,len(ok)),dtype=int)
 sheet=Image.new('RGB',(1280,780),'#222');draw=ImageDraw.Draw(sheet)
 for j,i in enumerate(selected):
  im=Image.open(out/'preview'/f"{ok[i]['frame']}.jpg");im.thumbnail((320,240));x=j%4*320;y=j//4*260;sheet.paste(im,(x,y));draw.text((x,y+240),ok[i]['frame'],fill='white')
 sheet.save(out/'contact_sheet.jpg')
 report=dict(sequence=str(a.sequence),poses=str(a.poses),mesh=str(a.mesh),stride=a.stride,start_sample=a.start_sample,depth_occlusion_filter=False,class_name='blue_kfs',exported=len(ok),skipped=len(records)-len(ok),records=records)
 (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
 (out/'README_中文.md').write_text(f'''# FP 纯模型投影 mask 试验

来源：{a.sequence}，位姿：{a.poses}。每 {a.stride} 帧取一张，类别 blue_kfs。

不读取实测深度，完全不启用 30mm 遮挡剔除。模型完整投影保存在 masks_projected，简化多边形反向栅格化 mask 在 masks，YOLO 归一化多边形在 labels。所有标注都是待复核伪标签，不自动加入现有训练集。

查看 contact_sheet.jpg 和 preview。打开 Labelme：

```bash
/home/tomato/code/box_seg_annotation/start_labelme.sh {out}/images --output {out}/labelme --config /home/tomato/code/box_seg_annotation/config/labelme.yaml
```

无深度筛选会保留被手遮挡的模型投影；位姿/模型尺寸有误时也不能自动纠正。人工修改 JSON 后，现有 labels/masks 不会自动同步，需要重新转换；不要直接当作可见区域真值。
''')
if __name__=='__main__':main()
