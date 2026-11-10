import sys, os, numpy as np, pybullet as p
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from attitude_env import AttitudeEnv
from PIL import Image
env = AttitudeEnv(rate_levels=(-90,-30,0,30,90))
env.reset(seed=0, options={'tilt_deg': [25.0, 0.0], 'rate_deg': [0.0, 0.0]})
cid, did = env.aviary.CLIENT, env.aviary.DRONE_IDS[0]
pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=cid)
ims = []
for yaw in (0, 90):
    for pitch in (8,):
        view = p.computeViewMatrixFromYawPitchRoll(pos, 0.18, yaw, pitch, 0, 2, physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(50, 1.2, 0.01, 10, physicsClientId=cid)
        img = np.reshape(p.getCameraImage(360, 300, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)[2], (300, 360, 4))[:, :, :3]
        ims.append(img.astype(np.uint8))
Image.fromarray(np.concatenate(ims, axis=1)).save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cam.png'))
print('ok')
