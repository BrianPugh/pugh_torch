import torch
import logging
log = logging.getLogger(__name__)

class LoadStateDictMixin:
    def load_state_dict(self, state_dict, strict=True):
        """Confirms and logs the weights that you expect are loaded when
        ``strict=False``.

        Returns
        -------
        changed : list of str
             Parameter names that were updated via loading the state_dict
        unchanged : list of str
             Parameter names that were NOT updated via loading the state_dict
        shape_mismatch : list of str
             Parameter names that were NOT updated via loading the state_dict
             due to the model's paramemeter shape not matching the state_dict.
             This is strictly a subset of ``unchanged``.
        """

        params = list(self.named_parameters())

        # Clone all the initial parameters so we can compare them to
        # the loaded ones
        initial_params = [(name, p.clone()) for (name, p) in params]

        shape_mismatch = []
        if not strict:
            # Torch has issues of loading weights whenever the shapes don't match
            # However, this is super common when fine-tuning a model.
            # So we work around this by modifying the loaded state_dict
            state_dict = state_dict.copy()
            for name, tensor in self.state_dict().items():
                if name in state_dict and state_dict[name].shape != tensor.shape:
                    shape_mismatch.append(name)
                    state_dict.pop(name)

        # Perform the loading
        super().load_state_dict(state_dict, strict=strict)

        # Compare all parameters
        changed, unchanged = [], []
        for initial, loaded in zip(initial_params, self.named_parameters()):
            name = initial[0]
            if torch.equal(initial[1], loaded[1]):
                unchanged.append(name)
            else:
                changed.append(name)

        if not strict:
            # Log a summary of the loading
            leader = "\n    "
            log_str = (
                "\n"
                + "Weights Loaded\n"
                + "--------------"
                + leader
                + leader.join(changed)
                + "\n\n"
                + "Weights NOT Loaded (or loaded values were identical to init)\n"
                + "------------------------------------------------------------"
                + leader
                + leader.join(unchanged)
                + "\n\n"
                + "Weights NOT Loaded Due to Shape Mismatch\n"
                + "----------------------------------------"
                + leader
                + leader.join(shape_mismatch)
                + "\n\n"
            )
            log.info(log_str)

        return changed, unchanged, shape_mismatch
