"""PAC-GS research code: proxy-controlled generative multiview for single-image Gaussian splats.

A 3D proxy of the person (the SAM 3D Body mesh, or an existing splat) is rendered as a depth video
along a camera orbit that starts exactly at the photo's camera. Wan 2.2 Fun-Control turns that depth
video plus the photo into a realistic orbit video whose frames follow the proxy pixel for pixel, so
every frame's camera is known by construction: no structure-from-motion, no keypoint registration.
The frames (and the photo) then train a splat with the v2 trainer (i2g.splat, kind "video").

Plans: next-stage-research/docs/pacgs_implementation_plan.md (the anchor experiments there are
parked; this package is the proxy-controlled video part). Run: python -m pacgs --help
"""
