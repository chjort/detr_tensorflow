import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects


class LearningRateMultiplier(tf.keras.optimizers.Optimizer):
    def __init__(
            self,
            optimizer,
            lr_multipliers,
            name: str = "LearningRateMultiplier",
            **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        if not isinstance(lr_multipliers, dict):
            raise TypeError("sequential_update must be of type dict")

        self._optimizer = optimizer
        self.lr_multipliers = lr_multipliers

        # TODO: Set only initial learning rate for layers???

    def _get_multiplier(self, param):
        max_key_len = 0
        max_len_key = None
        for k in self.lr_multipliers.keys():
            if k in param.name:
                if len(k) > max_key_len:
                    max_key_len = len(k)
                    max_len_key = k

        return self.lr_multipliers.get(max_len_key, 1)

    def _get_params_multipliers(self, params):
        """
        Returns:
            multiplier_params = {
                lr_mult_1: [param_i, ..., param_j],
                ...
                lr_mult_n: [param_k, ..., param_l],
            }
        """

        multiplier_params = {}
        for param in params:
            lr_mult = self._get_multiplier(param)
            multiplier_params.setdefault(lr_mult, []).append(param)
        return multiplier_params

    def get_updates(self, loss, params):
        multiplier_params = self._get_params_multipliers(params)

        updates = []
        base_lr = super().lr
        for multiplier, params_i in multiplier_params.items():
            super().lr = base_lr * multiplier
            updates.extend(super().get_updates(loss, params_i))

        super().lr = base_lr

        return updates

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name)

    def _resource_apply_dense(self, grad, var):
        return self._optimizer._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._optimizer._resource_apply_sparse(grad, var, indices)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        return self._optimizer._resource_apply_sparse_duplicate_indices(grad, var, indices)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            'lr_multipliers': self.lr_multipliers
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)


get_custom_objects().update({'LearningRateMultiplier': LearningRateMultiplier})
