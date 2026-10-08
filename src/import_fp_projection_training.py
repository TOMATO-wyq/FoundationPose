#!/usr/bin/env python3
"""导入指定 FP 伪标签批次；保留来源，整段录像只进入训练集。"""
import argparse,datetime,json,shutil
from pathlib import Path
import numpy as np
from ultralytics.data.converter import merge_multi_segment
ROOT=Path('/home/tomato/code/box_seg_annotation')
OLD=Path('/home/tomato/code/yolo_official/capture/fp_dynamic_trial_stride5')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--static',type=Path,required=True);p.add_argument('--dynamic',type=Path,required=True);a=p.parse_args()
 existing=json.loads((ROOT/'manifest.json').read_text())
 if any(it['group'] in ['fp_d435_dynamic_stride5','fp_d435_static_stride5'] for it in existing):raise ValueError('该批次已导入；请在 Labelme 编辑现有副本，不重复覆盖')
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 backup=ROOT/'import_backups'/stamp;backup.mkdir(parents=True)
 shutil.copy2(ROOT/'manifest.json',backup/'annotation_manifest.json')
 # 先完整备份旧派生数据，包含前 99 张人工修改和第 100 张以后的旧标签。
 shutil.copytree(OLD,backup/'fp_dynamic_trial_stride5',symlinks=True)
 old_manifest=json.loads((OLD/'manifest.json').read_text())
 new_manifest=json.loads((a.dynamic/'manifest.json').read_text())
 if new_manifest['start_sample']!=100 or new_manifest['stride']!=5:raise ValueError('本次仅允许从第 100 张、stride=5 的重处理结果')
 first99=old_manifest['records'][:99]
 for record in new_manifest['records']:
  if record['status']!='ok':raise ValueError('重处理有失败帧，不替换旧数据')
  frame=record['frame'];name=frame+'.png'
  for target,source in [('masks','masks'),('masks_projected','masks_projected'),('preview','preview'),('labels_seg','labels')]:
   ext='.jpg' if target=='preview' else '.txt' if target=='labels_seg' else '.png'
   shutil.copy2(a.dynamic/source/(frame+ext),OLD/target/(frame+ext))
  ann=json.loads((a.dynamic/'labelme'/(frame+'.json')).read_text());ann['imagePath']=name
  (OLD/'labelme'/(frame+'.json')).write_text(json.dumps(ann,indent=2,ensure_ascii=False))
  # 兼容旧目录的检测派生文件；实际训练仍只使用分割多边形。
  xy=np.asarray(ann['shapes'][0]['points']);lo=xy.min(0);hi=xy.max(0)+1;wh=np.array([ann['imageWidth'],ann['imageHeight']])
  box=np.r_[(lo+hi)/2/wh,(hi-lo)/wh]
  (OLD/'labels_bbox'/(frame+'.txt')).write_text('0 '+' '.join(f'{v:.8f}' for v in box)+'\n')
 old_manifest['records']=first99+new_manifest['records']
 old_manifest['reprocessing']=dict(start_sample=100,first_source_frame=new_manifest['records'][0]['frame'],depth_occlusion_filter=False,backup=str(backup))
 (OLD/'manifest.json').write_text(json.dumps(old_manifest,indent=2,ensure_ascii=False))
 (OLD/'README_中文.md').write_text(f'前 99 张保留已有 JSON；第 100 张起关闭深度遮挡剔除重新投影。\n旧版完整备份：{backup}\n前 99 张旧类别 BlueTrueKFS13 在训练导入副本中转换为 blue_kfs，原 JSON 不改。\n旧 contact_sheet.jpg 属于旧批次，请查看新批次 preview 或训练导出的 overlays。\n')
 rows=json.loads((ROOT/'manifest.json').read_text());known={it['id'] for it in rows};added=[];merged=[]
 for source,group in [(OLD,'fp_d435_dynamic_stride5'),(a.static,'fp_d435_static_stride5')]:
  for image in sorted((source/'images').glob('*.png')):
   key=group+'_'+image.stem
   if key in known:raise ValueError(f'已经导入：{key}')
   ann=json.loads((source/'labelme'/(image.stem+'.json')).read_text())
   shapes=ann['shapes']
   if not shapes:raise ValueError(f'空伪标签需要人工处理：{key}')
   # 旧筛选可能把同一实例切成多块；用官方转换连接轮廓，不虚增实例数。
   grouped={}
   for i,shape in enumerate(shapes):
    if shape['shape_type']!='polygon':raise ValueError('只接受 polygon')
    instance=shape.get('group_id');grouped.setdefault(('group',instance) if instance is not None else ('shape',i),[]).append(shape)
   converted=[]
   for parts in grouped.values():
    xy=np.asarray(parts[0]['points']) if len(parts)==1 else np.concatenate(merge_multi_segment([np.asarray(s['points']).ravel().tolist() for s in parts]))
    if len(parts)>1:merged.append(key)
    converted.append(dict(label='blue_kfs',points=xy.tolist(),shape_type='polygon',group_id=len(converted)+1,flags={}))
   ann['shapes']=converted;ann['imagePath']='../samples/'+key+'.png'
   ann['fp_source_annotation']=str(source/'labelme'/(image.stem+'.json'))
   shutil.copy2(image,ROOT/'samples'/(key+'.png'))
   (ROOT/'labelme_json'/(key+'.json')).write_text(json.dumps(ann,indent=2,ensure_ascii=False))
   row=dict(id=key,file=key+'.png',group=group,split='train',split_locked=True,training_approved=True,annotation_origin='FP pseudo label explicitly authorized by user',source_frame=int(image.stem),source=str(image))
   rows.append(row);added.append(row);known.add(key)
 (ROOT/'manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 report=dict(added=len(added),groups={g:sum(it['group']==g for it in added) for g in {it['group'] for it in added}},merged_disconnected_instances=merged,backup=str(backup),first99_preserved=True)
 (backup/'import_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
