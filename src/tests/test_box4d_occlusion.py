"""Occlusion, color-half joins and cadence regressions on rendered RGB-D."""
import unittest
from unittest.mock import patch

import numpy as np

from box4d.config import Config
from box4d.detector import Face
from box4d.pacing import DisplayPacer
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, world_box


class OcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def frame(self, cut=.04, size=.2, top=False):
        box = world_box((0,0,size/2),size,0,(0,-1.2,.1))
        box['red_axis_local'] = [0,0,1]
        a,b = (-.4,cut) if top else (cut,.4)
        occluder = dict(quad_camera=[[-.3,a,.85],[.3,a,.85],[.3,b,.85],[-.3,b,.85]])
        return self.renderer.render([box],depth_noise_m=.001,floor_height_camera_m=.1,occluders=[occluder])

    def cfg(self, **kwargs):
        return Config(up_camera=[0,-1,0],analyze_colors=True,two_color_sizes_m=(.2,),plane_grid_step=8,**kwargs)

    def test_bottom_quarter_recovered_from_current_depth_and_visible_edges(self):
        frame,truth,_ = self.frame()
        result,_ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(len(result['objects']),1)
        obj = result['objects'][0]
        np.testing.assert_allclose(obj['center_camera_m'],truth[0]['center_camera_m'],atol=.005)
        self.assertEqual(obj['method'],'depth_partial')
        self.assertEqual(obj['evidence'],'model_inferred')
        self.assertIsNone(obj['metrics']['reprojection_rms_px'])
        self.assertGreater(obj['metrics']['foreground_depth_fraction'],.1)
        self.assertEqual(obj['appearance']['orientation_status'],'red_up')
        self.assertEqual(obj['metrics']['refinement'],'no_consistent_correspondences')

    def test_half_hidden_pose_can_be_observable_while_color_direction_is_ambiguous(self):
        frame,truth,_ = self.frame(cut=0.)
        result,_ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(len(result['objects']),1)
        obj = result['objects'][0]
        np.testing.assert_allclose(obj['center_camera_m'],truth[0]['center_camera_m'],atol=.005)
        self.assertEqual(obj['appearance']['orientation_status'],'ambiguous')
        self.assertIsNone(obj['appearance']['red_axis_camera'])

    def test_background_missing_depth_wrong_size_and_top_occlusion_are_not_bottom_evidence(self):
        for kind in ('background','holes','wrong_size','top_hidden'):
            frame,_,instance = self.frame(size=.27 if kind=='wrong_size' else .2,top=kind=='top_hidden')
            if kind in ('background','holes'):
                mask = instance == -2
                frame.depth[mask] = 2.5 if kind=='background' else 0.
            result,_ = Pipeline(self.cfg()).process(frame)
            self.assertFalse(any(f['source']=='bottom_occlusion_model' for o in result['objects'] for f in o['faces']),kind)

    def test_roi_cut_and_disabled_recovery_do_not_return_partial_pose(self):
        frame,_,_ = self.frame()
        for cfg in (self.cfg(roi_xyxy=[250,180,390,270]),self.cfg(allow_bottom_occlusion=False)):
            result,_ = Pipeline(cfg).process(frame)
            self.assertEqual(result['objects'],[])

    def test_unresolved_size_and_pnp_only_do_not_guess_partial_pose(self):
        frame,_,_ = self.frame()
        for cfg in (self.cfg(sizes_m=(.2,.21)),self.cfg(method='pnp')):
            result,_ = Pipeline(cfg).process(frame)
            self.assertEqual(result['objects'],[])

    def test_wrong_size_silhouettes_and_floor_are_not_a_hidden_larger_cube(self):
        pipeline=Pipeline(self.cfg(color_line_fallback=True))
        for angle in (0,15,25,35,45,60,75):
            box=world_box((0,0,.135),.27,angle)
            box['red_axis_local']=[0,0,1]
            frame,_,_=self.renderer.render([box],depth_noise_m=.001)
            result,_=pipeline.process(frame)
            self.assertEqual(result['objects'],[],angle)

    def test_color_halves_join_without_complete_outline_proposal(self):
        box = world_box((0,0,.1),.2,0,(0,-1.2,.1))
        box['red_axis_local']=[0,0,1]
        frame,truth,_ = self.renderer.render([box],depth_noise_m=.001,floor_height_camera_m=.1)
        # Work directly in camera coordinates: the face is at optical z=1.1.
        quads = [np.array([[278,198],[362,198],[362,240],[278,240]],float),
                 np.array([[278,240],[362,240],[362,282],[278,282]],float)]
        faces = [Face(q,'paint_half',1.) for q in quads]
        pipeline = Pipeline(self.cfg())
        with patch.object(pipeline.detector,'propose',return_value=(faces,dict(proposal_count=2,detector_ms=0.),np.zeros(frame.depth.shape,np.uint8))):
            result,_ = pipeline.process(frame)
        self.assertEqual(len(result['objects']),1)
        obj=result['objects'][0]
        np.testing.assert_allclose(obj['center_camera_m'],truth[0]['center_camera_m'],atol=.005)
        self.assertEqual(obj['faces'][0]['source'],'paired_color_halves')
        # A single color stripe alone cannot initiate an entire cube.
        frame.rgb[:]=[220,25,25]
        with patch.object(pipeline.detector,'propose',return_value=(faces,dict(proposal_count=2,detector_ms=0.),np.zeros(frame.depth.shape,np.uint8))):
            result,_ = pipeline.process(frame)
        self.assertEqual(result['objects'],[])

    def test_small_colored_top_on_larger_cube(self):
        for angle in (0,15,30):
            camera=(0,-1.8,.35)
            base=world_box((0,0,.175),.35,angle,camera,(230,35,35))
            top=world_box((0,0,.45),.2,angle,camera)
            top['red_axis_local']=[0,0,1]
            frame,truth,_ = self.renderer.render([base,top],depth_noise_m=.001)
            result,_ = Pipeline(self.cfg()).process(frame)
            small=[o for o in result['objects'] if o['size_m']==.2]
            self.assertEqual(len(small),1,angle)
            np.testing.assert_allclose(small[0]['center_camera_m'],truth[1]['center_camera_m'],atol=.015)

    def test_coplanar_blue_base_merging_with_blue_lower_half(self):
        camera=(0,-1.8,.35)
        base=world_box((0,0,.175),.35,0,camera,(40,45,225))
        top=world_box((0,-.075,.45),.2,0,camera)
        top['red_axis_local']=[0,0,1]
        frame,truth,_=self.renderer.render([base,top],depth_noise_m=.001)
        result,_=Pipeline(self.cfg()).process(frame)
        small=[o for o in result['objects'] if o['size_m']==.2]
        self.assertEqual(len(small),1)
        np.testing.assert_allclose(small[0]['center_camera_m'],truth[1]['center_camera_m'],atol=.005)
        self.assertTrue(any(f['evidence'].get('support_contact') for f in small[0]['faces']))
        for obj in result['objects']:
            expected=truth[0] if obj['size_m']==.35 else truth[1]
            np.testing.assert_allclose(obj['center_camera_m'],expected['center_camera_m'],atol=.015)

    def test_target_color_filter_separates_geometry_from_target_identity(self):
        box=world_box((0,0,.1),.2,15,color=(155,155,155))
        frame,_,_=self.renderer.render([box],depth_noise_m=.001)
        generic,_=Pipeline(self.cfg()).process(frame)
        self.assertTrue(generic['objects'])
        filtered,_=Pipeline(self.cfg(require_target_colors=True)).process(frame)
        self.assertEqual(filtered['objects'],[])
        self.assertTrue(filtered['appearance_rejections'])
        # A real partly occluded target stays valid despite ambiguous color yaw.
        partial,_,_=self.frame(cut=0.)
        result,_=Pipeline(self.cfg(require_target_colors=True)).process(partial)
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['objects'][0]['appearance']['orientation_status'],'ambiguous')


class PacingTests(unittest.TestCase):
    def test_processing_jitter_does_not_reset_every_display_deadline(self):
        pacer=DisplayPacer(30.,0.)
        times=[(i+.5)/30+(.004 if i%2 else -.004) for i in range(120)]
        self.assertEqual(sum(pacer.ready(t) for t in times),120)
        self.assertFalse(pacer.ready(times[-1]+.001))
        self.assertTrue(pacer.ready(8.))  # skip old slots, no catch-up burst
        self.assertFalse(pacer.ready(8.001))


if __name__=='__main__':
    unittest.main()
