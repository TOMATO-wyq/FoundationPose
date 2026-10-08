"""三组深度滤波实验，按共同源帧与相同Tag参考评估；不删除异常。"""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_fp_apriltag import read_pose,cube_symmetries,angles,statistics
ROOT=Path('/home/tomato/6Dpose/outputs')
OUT=ROOT/'depth_filter_comparison'
OUT.mkdir(exist_ok=True)
groups={'raw':'d435i_dynamic_fp','5x5':'d435i_dynamic_fp_filtered','9x9':'d435i_dynamic_fp_filtered_large'}
manifest=list(csv.DictReader((ROOT/'d435i_dynamic_sequence/frames.csv').open()))
tag_rows={r['source_frame']:r for r in csv.DictReader((ROOT/'d435i_dynamic_tag_075/detections.csv').open())}
sym=cube_symmetries();transform=np.eye(4);transform[2,3]=-.175
rows=[];coverage={g:len(list((ROOT/p/'ob_in_cam').glob('*.txt'))) for g,p in groups.items()}
for frame in manifest:
 tag=tag_rows.get(frame['source_frame'])
 if tag is None or tag['status']!='detected':continue
 assert tag['timestamp_ns']==frame['timestamp_ns'] and tag['rgb_file']==frame['rgb_file']
 name=Path(frame['rgb_file']).with_suffix('.txt')
 paths={g:ROOT/p/'ob_in_cam'/name for g,p in groups.items()}
 if not all(p.exists() for p in paths.values()):continue
 reference=read_pose(ROOT/'d435i_dynamic_tag_075/tag_in_cam'/name)@transform
 for group,p in paths.items():
  fp=read_pose(p);delta=(fp[:3,3]-reference[:3,3])*1000
  angle=float(angles(np.swapaxes(sym,1,2)@(reference[:3,:3].T@fp[:3,:3])).min())
  rows.append(dict(group=group,source_frame=int(frame['source_frame']),time_s=(int(frame['timestamp_ns'])-int(manifest[0]['timestamp_ns']))/1e9,delta_x_mm=delta[0],delta_y_mm=delta[1],delta_z_mm=delta[2],center_difference_mm=np.linalg.norm(delta),sym_angle_deg=angle))
with (OUT/'per_frame.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=[]
for group in groups:
 rr=[r for r in rows if r['group']==group]
 for window in ['all','excluding_60_62s']:
  selected=rr if window=='all' else [r for r in rr if not 60<=r['time_s']<=62]
  for key in ['delta_x_mm','delta_y_mm','delta_z_mm','center_difference_mm','sym_angle_deg']:
   v=np.array([r[key] for r in selected]);m=statistics(v);m['mae']=float(np.abs(v).mean());m['abs_p95']=float(np.quantile(np.abs(v),.95))
   metrics.append(dict(group=group,window=window,metric=key,**m))
with (OUT/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
(OUT/'summary.json').write_text(json.dumps(dict(coverage=coverage,common_valid_frames=len(rows)//3,metrics=metrics),indent=2))
fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
for group,color in [('raw','#444444'),('5x5','#0072B2'),('9x9','#D55E00')]:
 rr=[r for r in rows if r['group']==group]
 for ax,key in zip(axes,['delta_x_mm','delta_y_mm','delta_z_mm','center_difference_mm']):
  ax.plot([r['time_s'] for r in rr],[r[key] for r in rr],label=group,color=color,lw=.9,alpha=.8)
  ax.set_ylabel(key.replace('_',' '));ax.grid(alpha=.2)
axes[0].legend(ncol=3);axes[-1].set_xlabel('Source time (s)')
fig.suptitle('Same-frame depth filter comparison: FP minus tag-derived reference')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
for m in metrics:
 if m['window']=='all':print(m['group'],m['metric'],'MAE',round(m['mae'],3),'RMSE',round(m['rmse'],3),'absP95',round(m['abs_p95'],3))
print('Coverage:',coverage,'common valid',len(rows)//3)
