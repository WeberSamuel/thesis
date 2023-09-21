from stable_baselines3.common.callbacks import BaseCallback
    
class TagExplorationDataCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            info["is_exploration"] = True
        return super()._on_step()