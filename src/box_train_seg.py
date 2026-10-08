#!/usr/bin/env python3
"""紫色箱体 YOLO 实例分割试训；源码在 6Dpose/src，数据和结果保存在独立目录。"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,math,os,shutil,sys,time
ROOT=Path('/home/tomato/code/box_seg_annotation')
DEFAULT_NEGATIVES=Path(__file__).resolve().parent/'config'/'box_confirmed_negative_ids.txt'
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'config/matplotlib'))

def args_parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--include-saved-group',action='append',default=[],help='明确使用该组保存但未勾 reviewed 的标注；不修改原 JSON')
 negatives=p.add_mutually_exclusive_group()
 negatives.add_argument('--confirmed-negative-ids',type=Path,help='已确认没有 blue_kfs 的未标注图片 ID 清单，每行一个；新训练默认读取项目内确认清单')
 negatives.add_argument('--no-confirmed-negatives',action='store_true',help='本次新训练不追加确认清单中的负样本')
 p.add_argument('--epochs',type=int,default=150);p.add_argument('--patience',type=int,default=0,help='0 关闭早停，默认完整训练指定轮数；正数为早停等待轮数')
 p.add_argument('--imgsz',type=int,default=640);p.add_argument('--batch',type=int,default=2)
 p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='0');p.add_argument('--seed',type=int,default=42)
 p.add_argument('--val-fraction',type=float,default=.2);p.add_argument('--gap',type=int,default=10,help='同一拍摄组连续块间丢弃的已标注样本数')
 p.add_argument('--split-mode',choices=['temporal','manifest'],default='temporal',help='temporal 粗分连续块；manifest 使用已有独立片段划分')
 p.add_argument('--model',type=Path,default=ROOT/'models/yolo11n-seg.pt')
 p.add_argument('--name',default=None);p.add_argument('--prepare-only',action='store_true')
 p.add_argument('--resume',type=Path,help='从 last.pt 恢复中断训练')
 p.add_argument('--preview-count',type=int,default=12)
 return p

def resolve_negative_ids(a):
 if not a.resume and not a.no_confirmed_negatives and a.confirmed_negative_ids is None and DEFAULT_NEGATIVES.exists():
  a.confirmed_negative_ids=DEFAULT_NEGATIVES
 return a

def mixed_validation_window(items,fraction):
 """Smallest contiguous validation block with both class quotas; latest tie wins."""
 negative=[bool(it.get('confirmed_negative')) for it in items]
 total=collections.Counter(negative)
 wanted={kind:max(1,math.ceil(total[kind]*fraction)) for kind in (False,True)}
 counts=collections.Counter();left=0;best=None
 for right,kind in enumerate(negative):
  counts[kind]+=1
  if any(counts[k]<wanted[k] for k in wanted):continue
  while counts[negative[left]]>wanted[negative[left]]:
   counts[negative[left]]-=1;left+=1
  candidate=(left,right+1)
  if best is None or candidate[1]-candidate[0]<=best[1]-best[0]:best=candidate
 if best is None:raise ValueError('无法产生同时包含正负样本的连续验证片段')
 return best

def split_rows(a):
 rows=json.loads((ROOT/'manifest.json').read_text());eligible=[];missing=0;unconfirmed=0
 negative_file=getattr(a,'confirmed_negative_ids',None)
 negatives=set()
 if negative_file:
  lines=[line.strip() for line in negative_file.read_text().splitlines() if line.strip()]
  if len(lines)!=len(set(lines)):raise ValueError('负样本 ID 清单含重复项')
  negatives=set(lines);known={it['id'] for it in rows}
  if not negatives:raise ValueError('负样本 ID 清单为空')
  if negatives-known:raise ValueError(f'负样本 ID 不在 manifest 中：{sorted(negatives-known)}')
  for key in negatives:
   if Path(key).name!=key or (ROOT/'labelme_json'/f'{key}.json').exists():raise ValueError(f'{key} 不是未标注图片；已有 JSON 请按 reviewed 流程使用')
 for it in rows:
  p=ROOT/'labelme_json'/f"{it['id']}.json"
  if not p.exists():
   missing+=1
   if it['id'] in negatives:eligible.append(dict(it,confirmed_negative=True))
   continue
  data=json.loads(p.read_text())
  if data.get('flags',{}).get('reviewed') is not True and it['group'] not in a.include_saved_group and not it.get('training_approved',False):
   unconfirmed+=1;continue
  eligible.append(dict(it))
 if not eligible:raise ValueError('没有已确认标签；按用户明确声明可传 --include-saved-group')
 discarded=[];result=[]
 groups=collections.defaultdict(list)
 for it in eligible:groups[it['group']].append(it)
 existing_val=any(it['split']=='val' for it in eligible)
 for group,items in groups.items():
  items.sort(key=lambda it:(it.get('source_frame',0),it['id']))
  # 有独立验证片段时保留原有划分；少量旧演示图不拆分。
  if a.split_mode=='manifest' or existing_val or len(items)<30 or all(it.get('split_locked',False) for it in items):
   result.extend(items);continue
  if any(it.get('confirmed_negative') for it in items) and any(not it.get('confirmed_negative') for it in items):
   lo,hi=mixed_validation_window(items,a.val_fraction)
   for i,it in enumerate(items):
    it['source_group']=group
    if max(0,lo-a.gap)<=i<lo or hi<=i<min(len(items),hi+a.gap):discarded.append(it['id']);continue
    it['split']='val' if lo<=i<hi else 'train'
    it['group']=group+'_temporal_'+it['split'];it['split_method']='mixed_contiguous_validation_block_with_two_sided_gap'
    result.append(it)
   continue
  nval=max(1,math.ceil(len(items)*a.val_fraction));cut=len(items)-nval;train_end=cut-a.gap
  if train_end<1:raise ValueError(f'{group} 样本不足以保留 {a.gap} 张隔离带')
  for i,it in enumerate(items):
   it['source_group']=group
   if train_end<=i<cut:discarded.append(it['id']);continue
   it['split']='train' if i<train_end else 'val'
   # 记录同一次拍摄的连续子片段；不是伪称独立拍摄场景。
   it['group']=group+'_temporal_'+it['split'];it['split_method']='same_capture_contiguous_blocks_with_gap'
   result.append(it)
 included_saved=[it['group'] for it in result if it.get('source_group',it['group']) in a.include_saved_group or it.get('training_approved',False)]
 report=dict(method=a.split_mode,val_fraction=a.val_fraction,gap=a.gap,counts=dict(collections.Counter(it['split'] for it in result)),discarded_gap_ids=discarded,missing_json=missing,unconfirmed_skipped=unconfirmed,independent_validation=existing_val,risk='同一拍摄场景的连续留出，仍有场景泄漏，验证指标仅用于粗略试训' if not existing_val else '独立拍摄片段划分')
 report['confirmed_negative_ids']=[it['id'] for it in result if it.get('confirmed_negative')]
 report['confirmed_negative_counts']=dict(collections.Counter(it['split'] for it in result if it.get('confirmed_negative')))
 if not any(it['split']=='val' for it in result):raise ValueError('验证集为空：请加入独立片段或用 temporal 粗分模式')
 return result,sorted(set(included_saved)),report

def validate_negative_images(rows):
 """Check cross-split duplicate leakage before exporting any new dataset."""
 if not any(it.get('confirmed_negative') for it in rows):return
 hashes={}
 for it in rows:
  file=ROOT/'samples'/it['file'];digest=hashlib.sha256(file.read_bytes()).hexdigest()
  if digest in hashes and hashes[digest]!=it['split']:raise ValueError('负样本/正样本之间存在跨集合重复图片')
  hashes[digest]=it['split']

def append_confirmed_negatives(dataset,rows,report,negative_file):
 """Empty labels only for explicitly confirmed IDs; never rewrite source JSON."""
 import cv2,numpy as np
 from ultralytics.data.utils import verify_image_label
 for it in rows:
  if not it.get('confirmed_negative'):continue
  key,split=it['id'],it['split'];source=ROOT/'samples'/it['file']
  image=cv2.imread(str(source))
  if image is None:raise ValueError(f'负样本无法读取：{source}')
  target=dataset/'images'/split/f'{key}.png';label=dataset/'labels'/split/f'{key}.txt'
  if target.exists() or label.exists():raise ValueError(f'负样本会覆盖已有导出：{key}')
  shutil.copy2(source,target);label.write_text('')
  h,w=image.shape[:2]
  cv2.imwrite(str(dataset/'masks'/split/f'{key}.png'),np.zeros((h,w),np.uint8))
  cv2.imwrite(str(dataset/'overlays'/split/f'{key}.jpg'),image)
  result=verify_image_label((str(target),str(label),'',False,1,0,0,False))
  if result[0] is None or len(result[3])!=0:raise ValueError(f'{key} 官方负样本 loader 验证失败：{result[-1]}')
  annotation=dict(shapes=[],flags=dict(reviewed=True),imageWidth=w,imageHeight=h,
                  imagePath=str(source),confirmation='explicit_negative_id_list',confirmation_file=str(negative_file))
  (dataset/'source_annotations'/f'{key}.json').write_text(json.dumps(annotation,indent=2,ensure_ascii=False))
  report['counts'][split]+=1
 report['confirmed_negative_ids']=[it['id'] for it in rows if it.get('confirmed_negative')]
 (dataset/'split_manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 (dataset/'export_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 if negative_file:shutil.copy2(negative_file,dataset/'confirmed_negative_ids.txt')

def previews(weights,dataset,run_dir,count,device,imgsz):
 import cv2,numpy as np
 from ultralytics import YOLO
 model=YOLO(str(weights));images=sorted((dataset/'images/val').glob('*.png'))
 out=run_dir/'predictions_val';out.mkdir(exist_ok=True);records=[]
 for i in np.linspace(0,len(images)-1,min(count,len(images)),dtype=int):
  p=images[i];r=model.predict(str(p),imgsz=imgsz,device=device,conf=.25,retina_masks=True,verbose=False)[0]
  cv2.imwrite(str(out/(p.stem+'.jpg')),r.plot())
  mask=np.zeros(r.orig_shape,np.uint8)
  if r.masks is not None:
   indices=[j for j,c in enumerate(r.boxes.cls.tolist()) if int(c)==0]
   if indices:
    j=max(indices,key=lambda j:float(r.boxes.conf[j]));mask=(r.masks.data[j].cpu().numpy()>.5).astype(np.uint8)*255
  cv2.imwrite(str(out/(p.stem+'_mask.png')),mask)
  records.append(dict(image=p.name,detections=len(r.boxes),confidence=r.boxes.conf.tolist()))
 (out/'prediction_report.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))

def main():
 a=resolve_negative_ids(args_parser().parse_args())
 if a.epochs<1 or a.patience<0 or a.batch<1 or a.gap<0 or not 0<a.val_fraction<.5:raise SystemExit('请检查 epochs/patience/batch/gap/val-fraction（patience=0 关闭早停）')
 if shutil.disk_usage(ROOT).free<3*1024**3:raise SystemExit('至少需要 3GB 剩余空间')
 import torch,ultralytics
 if ultralytics.__version__!='8.3.161':raise SystemExit('请使用已配置的独立 env（Ultralytics 8.3.161）')
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
 if a.resume:
  if a.confirmed_negative_ids:raise SystemExit('新增负样本应开启新训练，不能改 resume 的旧数据快照')
  ckpt=torch.load(a.resume,map_location='cpu',weights_only=False);dataset=Path(ckpt['train_args']['data']).parent
  split_report=json.loads((dataset/'split_report.json').read_text())
 else:
  sys.path.insert(0,str(ROOT));from export_labelme import export
  rows,include,split_report=split_rows(a);dataset=ROOT/'training_datasets'/f'box_seg_{stamp}'
  validate_negative_images(rows)
  export_report=export(output_dir=dataset,include_saved_groups=include,manifest_rows=[it for it in rows if not it.get('confirmed_negative')],ignore_other_labels=True)
  append_confirmed_negatives(dataset,rows,export_report,a.confirmed_negative_ids)
  for split in ('train','val'):
   if not any(p.read_text().strip() for p in (dataset/'labels'/split).glob('*.txt')):raise ValueError(f'{split} 全为负样本，不能用于本次分割训练/验证')
  (dataset/'split_report.json').write_text(json.dumps(split_report,indent=2,ensure_ascii=False))
  print(json.dumps(dict(dataset=str(dataset),split=split_report,instances=export_report['instances']),indent=2,ensure_ascii=False),flush=True)
  if a.prepare_only:return
 if not torch.cuda.is_available():raise SystemExit('未检测到 CUDA，请在宿主机用项目 env 运行')
 print('CUDA:',torch.cuda.get_device_name(0),'free/total MiB:',[round(v/1024**2) for v in torch.cuda.mem_get_info()],flush=True)
 if not a.resume and not a.model.is_file():raise SystemExit(f'缺少官方分割权重：{a.model}')
 from ultralytics.models.yolo.segment.train import SegmentationTrainer
 # 训练过程的辅助下载也留在独立目录。
 os.chdir(ROOT)
 if a.resume:trainer=SegmentationTrainer(overrides=dict(model=str(a.resume),resume=str(a.resume),device=a.device))
 else:
  name=a.name or f'box_seg_trial_{stamp}'
  if Path(name).name!=name or (ROOT/'runs'/name).exists():raise SystemExit('运行名称无效或已存在；请换 --name 或使用 --resume')
  # 标准 Ultralytics EarlyStopping 根据验证 fitness：分割 mAP50-95 权重 0.9，mAP50 权重 0.1；框和 mask 两项求和。
  trainer=SegmentationTrainer(overrides=dict(model=str(a.model),data=str(dataset/'data.yaml'),epochs=a.epochs,patience=a.patience,imgsz=a.imgsz,batch=a.batch,device=a.device,workers=a.workers,seed=a.seed,deterministic=True,project=str(ROOT/'runs'),name=name,exist_ok=False,val=True,plots=True,amp=True,cache=False,save=True,save_period=25,optimizer='AdamW',lr0=.001,cos_lr=True,close_mosaic=10,hsv_h=.005,hsv_s=.3,hsv_v=.3))
 run_dir=Path(trainer.save_dir)
 (run_dir/'trial_info.json').write_text(json.dumps(dict(split=split_report,dataset=str(dataset),max_epochs=a.epochs,patience=a.patience,model=str(a.model),started_at=stamp),indent=2,ensure_ascii=False))
 start=time.time();trainer.train()
 weights=run_dir/'weights/best.pt'
 if not weights.exists():weights=run_dir/'weights/last.pt'
 previews(weights,dataset,run_dir,a.preview_count,a.device,a.imgsz)
 history=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader((run_dir/'results.csv').open())]
 def fitness(row):return sum(.1*row[f'metrics/mAP50({task})']+.9*row[f'metrics/mAP50-95({task})'] for task in ['B','M'])
 best_epoch=int(max(history,key=fitness)['epoch'])
 summary=dict(weights=str(weights),best_epoch=best_epoch,completed_epochs=trainer.epoch+1,max_epochs=trainer.epochs,early_stopped=trainer.epoch+1<trainer.epochs,elapsed_seconds=round(time.time()-start,1),metrics=trainer.metrics,split_warning=split_report['risk'],predictions=str(run_dir/'predictions_val'))
 (run_dir/'trial_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
