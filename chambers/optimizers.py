from typing import Union, Callable

from tensorflow.keras.utils import get_custom_objects
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.utils.types import FloatTensorLike


class LearningRateMultiplierMixin:
    def _get_multiplier(self, param):
        # TODO: Do this differently, in a way that prioritizes longest common prefix between param.name and key
        for k in self.lr_multipliers.keys():
            if k in param.name:
                return self.lr_multipliers[k]

    def get_updates(self, loss, params):
        super_opt = super(LearningRateMultiplierMixin, self)
        mult_lr_params = {p: self._get_multiplier(p) for p in params
                          if self._get_multiplier(p)}
        base_lr_params = [p for p in params if self._get_multiplier(p) is None]

        updates = []
        base_lr = super_opt.lr
        for param, multiplier in mult_lr_params.items():
            super_opt.lr = base_lr * multiplier
            updates.extend(super_opt.get_updates(loss, [param]))

        super_opt.lr = base_lr
        updates.extend(super_opt.get_updates(loss, base_lr_params))

        return updates

    def get_config(self):
        config = {'lr_multipliers': self.lr_multipliers}
        base_config = super(LearningRateMultiplierMixin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdamWLearningRateMultiplier(AdamW):
    def __init__(self,
                 weight_decay: Union[FloatTensorLike, Callable],
                 learning_rate: Union[FloatTensorLike, Callable] = 0.001,
                 beta_1: Union[FloatTensorLike, Callable] = 0.9,
                 beta_2: Union[FloatTensorLike, Callable] = 0.999,
                 epsilon: FloatTensorLike = 1e-07,
                 amsgrad: bool = False,
                 lr_multipliers=None,
                 name: str = "AdamW",
                 **kwargs):
        self.lr_multipliers = lr_multipliers or {}
        super(AdamWLearningRateMultiplier, self).__init__(weight_decay=weight_decay,
                                                          learning_rate=learning_rate,
                                                          beta_1=beta_1,
                                                          beta_2=beta_2,
                                                          epsilon=epsilon,
                                                          amsgrad=amsgrad,
                                                          name=name,
                                                          **kwargs)

    def _get_multiplier(self, param):
        # TODO: Do this differently, in a way that prioritizes longest common prefix between param.name and key
        for k in self.lr_multipliers.keys():
            if k in param.name:
                return self.lr_multipliers[k]

    def get_updates(self, loss, params):
        super_opt = super(AdamWLearningRateMultiplier, self)
        mult_lr_params = {p: self._get_multiplier(p) for p in params
                          if self._get_multiplier(p)}
        base_lr_params = [p for p in params if self._get_multiplier(p) is None]

        updates = []
        base_lr = super_opt.lr
        for param, multiplier in mult_lr_params.items():
            super_opt.lr = base_lr * multiplier
            updates.extend(super_opt.get_updates(loss, [param]))

        super_opt.lr = base_lr
        updates.extend(super_opt.get_updates(loss, base_lr_params))

        return updates

    def get_config(self):
        config = {'lr_multipliers': self.lr_multipliers}
        base_config = self.get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'AdamWLearningRateMultiplier': AdamWLearningRateMultiplier})
