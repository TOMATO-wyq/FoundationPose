"""Typed geometry rejections that permit specific recovery paths."""


class FaceSizeMismatch(ValueError):
    """Measured face dimensions match none of the configured square sizes."""


class TrackingPoseRejected(ValueError):
    """当前姿态未通过验收；结构化原因与异常文本使用同一来源。"""

    def __init__(self, stage, *checks):
        self.stage = stage
        self.checks = list(checks)
        super().__init__(", ".join(checks))
