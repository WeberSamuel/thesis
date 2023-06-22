from src.envs.toy_goal_env import ToyGoalEnv
from src.envs.samplers import RandomBoxSampler

def test_render():
    env = ToyGoalEnv(RandomBoxSampler(), render_mode="rgb_array")
    env.reset()
    img = env.render()
    assert img is not None
    assert img.shape == (256, 256, 3)
    env.close()