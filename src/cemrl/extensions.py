from typing import List

class CEMRLExtension:
    def _init_extension(self, model):
        self.model = model
        self.logger = model.logger

    def after_reconstruction_class_loss(self, locals):
        pass

    def after_reconstruction_loss_calculation(self, locals):
        pass

    def after_reconstruction_backward(self, locals):
        pass

    def after_reconstruction_step(self, locals):
        pass

class CEMRLExtensionList(CEMRLExtension):
    def __init__(self, extensions: List[CEMRLExtension]) -> None:
        self.extensions = extensions

    def _init_extension(self, model):
        for extension in self.extensions:
            extension._init_extension(model)