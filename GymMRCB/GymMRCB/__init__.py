from gym.envs.registration import register

register(
    id='MRCB-v0',
    entry_point='GymMRCB.envs:MRCB',
)