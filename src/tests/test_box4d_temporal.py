"""Current-image/depth tracking, including disappearance and detector outages."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import Detector
from box4d.edges import rgb_depth_edges
from box4d.geometry import Camera, angle_difference, yaw_from_pose, upright_from_camera
from box4d.io import Frame
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline
from box4d.tracking import Tracker


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer=Renderer(default_camera())

    def cfg(self,**kwargs):
        return Config(up_camera=[0,-1,0],analyze_colors=True,two_color_sizes_m=(.2,),
                      require_target_colors=True,temporal_recovery=True,plane_grid_step=8,**kwargs)

    def frame(self,i=0,x=0.,y=0.,yaw=15.):
        box=world_box((x,y,.1),.2,yaw)
        box['red_axis_local']=[0,0,1]
        return self.renderer.render([box],i,depth_noise_m=.001)

    def empty_detection(self,frame):
        return dict(sequence=frame.sequence,stamp_ns=frame.stamp_ns,calibration_id=frame.camera.calibration_id,
                    objects=[],rejected_faces=[],suppressed_hypotheses=[],appearance_rejections=[],
                    timing=dict(detector_ms=0.,solve_ms=0.,refinement_ms=0.,appearance_ms=0.,proposal_count=0)),np.zeros(frame.depth.shape,np.uint8)

    def seeded(self):
        p=TemporalPipeline(self.cfg())
        f,_,_=self.frame()
        result,_=p.process(f)
        self.assertEqual(len(result['objects']),1)
        self.assertTrue(p.states)
        return p

    def test_current_features_and_depth_bridge_detector_failures_during_motion(self):
        p=self.seeded();u=upright_from_camera([0,-1,0])
        with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
            for i in range(1,13):
                frame,truth,_=self.frame(i,x=i*.0015,y=i*.0005,yaw=15+i*.15)
                result,_=p.process(frame)
                self.assertEqual(len(result['objects']),1,(i,result.get('temporal')))
                obj=result['objects'][0]
                self.assertEqual(obj['source'],'current_rgbd_flow')
                self.assertEqual(obj['method'],'flow_pnp')
                np.testing.assert_allclose(obj['center_camera_m'],truth[0]['center_camera_m'],atol=.012)
                expected=yaw_from_pose(np.asarray(truth[0]['T_camera_from_box']),u)
                self.assertLess(abs(angle_difference(obj['yaw_u_rad'],expected)),np.deg2rad(4))
                self.assertGreaterEqual(obj['metrics']['tracked_features'],4)

    def test_removed_target_blank_rgb_and_missing_depth_stop_immediately(self):
        for kind in ('removed','blank_rgb','missing_depth'):
            with self.subTest(kind=kind):
                p=self.seeded();frame,_,_=self.frame(1)
                if kind=='removed':frame,_,_=self.renderer.render([],1)
                elif kind=='blank_rgb':frame.rgb[:]=0
                else:frame.depth[:]=0
                with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
                    result,_=p.process(frame)
                self.assertEqual(result['objects'],[],kind)
                self.assertEqual(result['temporal']['stale_poses_published'],0)

    def test_timestamp_gap_and_calibration_change_reset_tracking(self):
        for kind in ('gap','calibration'):
            p=self.seeded();frame,_,_=self.frame(1)
            if kind=='gap':frame.stamp_ns+=1_000_000_000
            else:
                k=frame.camera.k.copy();k[0,0]+=1
                frame.camera=Camera(frame.camera.width,frame.camera.height,k,frame.camera.d)
            with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
                result,_=p.process(frame)
            self.assertEqual(result['objects'],[],kind)

    def test_single_depth_failure_has_no_output_and_can_recover_next_frame(self):
        p=self.seeded();bad,_,_=self.frame(1);bad.depth[:]=0
        with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
            result,_=p.process(bad)
            self.assertEqual(result['objects'],[])
            self.assertGreater(result['temporal']['pending_feature_tracks'],0)
            good,truth,_=self.frame(2)
            recovered,_=p.process(good)
        self.assertEqual(len(recovered['objects']),1)
        self.assertEqual(recovered['objects'][0]['source'],'current_rgbd_flow')
        np.testing.assert_allclose(recovered['objects'][0]['center_camera_m'],truth[0]['center_camera_m'],atol=.012)

    def test_repeated_source_frame_is_rejected(self):
        p=self.seeded();frame,_,_=self.frame()
        with self.assertRaises(ValueError):p.process(frame)

    def test_detector_and_tracking_keep_value_edges_with_equal_gray_colors(self):
        camera=default_camera()
        rgb=np.full((camera.height,camera.width,3),(0,130,0),np.uint8)
        rgb[100:300,150:350]=(255,0,0)
        frame=Frame(rgb,np.ones(rgb.shape[:2],np.float32),camera,1)
        cfg=self.cfg(plane_proposals=False,canny_low=41,canny_high=255)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.assertFalse(cv2.Canny(gray,cfg.canny_low,cfg.canny_high).any())
        _,_,detector_edges=Detector(cfg).propose(frame)
        self.assertTrue(detector_edges.any())
        with patch.object(TemporalPipeline,'_full_detect',side_effect=lambda f,covered:(self.empty_detection(f)[0],None,[],(None,()))):
            pipeline=TemporalPipeline(cfg,live=True)
            try:
                _,tracking_edges=pipeline.process(frame)
            finally:
                pipeline.close()
        np.testing.assert_array_equal(tracking_edges,detector_edges)

    def test_tracking_edges_exclude_out_of_range_depth_and_respect_roi(self):
        camera=default_camera();rgb=np.zeros((camera.height,camera.width,3),np.uint8)
        depth=np.ones(rgb.shape[:2],np.float32);depth[:,320:]=5.
        frame=Frame(rgb,depth,camera,1)
        cfg=self.cfg(roi_xyxy=[100,100,400,400],max_depth_m=3.)
        self.assertFalse(rgb_depth_edges(frame,cfg)[2].any())
        frame.depth[:,320:]=1.2
        edges=rgb_depth_edges(frame,cfg)[2]
        self.assertTrue(edges[150:350,320].any())
        self.assertFalse(edges[:100].any())
        self.assertFalse(edges[400:].any())

    def test_current_validation_rejection_is_not_reported_as_no_target(self):
        pipeline=self.seeded();frame,_,_=self.frame(1);frame.depth[:]=0
        with patch.object(pipeline.detector,'process',side_effect=self.empty_detection),patch.object(pipeline,'_search_uncovered_colors',return_value=([],[])):
            result,_=pipeline.process(frame)
        self.assertEqual(result['objects'],[])
        self.assertEqual(result['frame_status'],'partial_or_rejected')
        Tracker(pipeline.cfg).update(result)
        self.assertEqual(result['frame_status'],'partial_or_rejected')

    def test_sparse_harris_face_can_reseed_and_still_reject_disappearance(self):
        original=cv2.goodFeaturesToTrack
        def sparse_harris(*args,**kwargs):
            return None if kwargs.get('useHarrisDetector') else original(*args,**kwargs)
        for kind in ('moving','removed','blank_rgb','missing_depth'):
            with self.subTest(kind=kind):
                p=TemporalPipeline(self.cfg())
                with patch('box4d.temporal.cv2.goodFeaturesToTrack',side_effect=sparse_harris):
                    first,_,_=self.frame()
                    initial,_=p.process(first)
                self.assertEqual(len(initial['objects']),1)
                self.assertEqual(p.states[0]['feature_method'],'shi_tomasi')
                current,truth,_=self.frame(1,x=.002)
                if kind=='removed':current,_,_=self.renderer.render([],1)
                elif kind=='blank_rgb':current.rgb[:]=0
                elif kind=='missing_depth':current.depth[:]=0
                with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
                    result,_=p.process(current)
                if kind=='moving':
                    self.assertEqual(len(result['objects']),1,result['temporal'])
                    self.assertEqual(result['objects'][0]['metrics']['feature_seed_method'],'shi_tomasi')
                    np.testing.assert_allclose(result['objects'][0]['center_camera_m'],truth[0]['center_camera_m'],atol=.012)
                else:
                    self.assertEqual(result['objects'],[])
                    self.assertTrue(result['temporal']['tracking_failure_details'])
                    self.assertEqual(result['temporal']['tracking_failure_details'][0]['size_m'],.2)

    def test_feature_method_can_keep_legacy_harris_or_force_shi_tomasi(self):
        frame,_,_=self.frame()
        for method in ('harris','shi_tomasi'):
            with self.subTest(method=method):
                p=TemporalPipeline(self.cfg(track_feature_method=method))
                p.process(frame)
                self.assertEqual(p.states[0]['feature_method'],method)
        with self.assertRaises(ValueError):self.cfg(track_feature_method='unknown').validate()

    def test_depth_border_contamination_does_not_reject_valid_face_interior(self):
        for inset in (0.,.10):
            with self.subTest(inset=inset):
                p=TemporalPipeline(self.cfg(track_depth_inset_fraction=inset))
                first,_,_=self.frame()
                p.process(first)
                current,truth,_=self.frame(1)
                quad=current.camera.project(
                    p.states[0]['quad_model'] @ p.states[0]['pose'][:3,:3].T+p.states[0]['pose'][:3,3])
                mask=np.zeros(current.depth.shape,np.uint8)
                cv2.fillConvexPoly(mask,np.rint(quad).astype(np.int32),255)
                width=min(np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1))
                distance=cv2.distanceTransform(mask,cv2.DIST_L2,3)
                ring=(mask>0)&(distance<width*.15)
                current.depth[ring]+=.20
                with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
                    result,_=p.process(current)
                if inset==0:
                    self.assertEqual(result['objects'],[])
                    failure=result['temporal']['tracking_failure_details'][0]
                    self.assertLess(failure['plane_support'],failure['plane_support_min'])
                else:
                    self.assertEqual(len(result['objects']),1,result['temporal'])
                    np.testing.assert_allclose(result['objects'][0]['center_camera_m'],truth[0]['center_camera_m'],atol=.012)

    def test_depth_inset_does_not_accept_two_planes_inside_the_face(self):
        p=self.seeded();current,_,_=self.frame(1)
        center=current.camera.project(p.states[0]['pose'][:3,3].reshape(1,3))[0,0]
        current.depth[:,int(center):]+=.15
        with patch.object(p.detector,'process',side_effect=self.empty_detection),patch.object(p,'_search_uncovered_colors',return_value=([],[])):
            result,_=p.process(current)
        self.assertEqual(result['objects'],[])

    def test_async_detection_is_revalidated_in_current_frame(self):
        for kind in ('moving','removed','delayed_moving','delayed_removed','delayed_missing_depth','too_old'):
            with self.subTest(kind=kind):
                p=TemporalPipeline(self.cfg(),live=True)
                try:
                    first,_,_=self.frame()
                    initial,_=p.process(first)
                    self.assertEqual(initial['objects'],[])
                    with p.worker_condition:
                        self.assertTrue(p.worker_condition.wait_for(lambda:p.worker_completed is not None or p.worker_error is not None,timeout=3.))
                    current,truth,_=self.frame(1,x=.002)
                    if kind in ('removed','delayed_removed'):current,_,_=self.renderer.render([],1)
                    if kind.startswith('delayed_'):current.stamp_ns+=500_000_000
                    if kind=='delayed_missing_depth':current.depth[:]=0
                    if kind=='too_old':current.stamp_ns+=1_000_000_000
                    result,_=p.process(current)
                    self.assertEqual(result['stamp_ns'],current.stamp_ns)
                    self.assertTrue(result['temporal']['async_detection'])
                    if kind in ('moving','delayed_moving'):
                        self.assertEqual(len(result['objects']),1)
                        obj=result['objects'][0]
                        self.assertEqual(obj['source'],'current_rgbd_flow')
                        self.assertEqual(obj['detector_seed_stamp_ns'],first.stamp_ns)
                        np.testing.assert_allclose(obj['center_camera_m'],truth[0]['center_camera_m'],atol=.012)
                    else:
                        self.assertEqual(result['objects'],[])
                        self.assertGreater(result['temporal']['discarded_old_detections'],0)
                        if kind=='too_old':
                            self.assertGreater(result['temporal']['expired_detections'],0)
                        else:
                            self.assertTrue(result['temporal']['detector_seed_failures'])
                finally:p.close()

    def test_local_color_region_can_acquire_when_full_scene_detector_misses(self):
        p=TemporalPipeline(self.cfg());frame,truth,_=self.frame()
        with patch.object(p.detector,'process',side_effect=self.empty_detection):
            result,_=p.process(frame)
        self.assertEqual(len(result['objects']),1)
        self.assertTrue(result['temporal']['local_searches'])
        np.testing.assert_allclose(result['objects'][0]['center_camera_m'],truth[0]['center_camera_m'],atol=.012)


if __name__=='__main__':unittest.main()
