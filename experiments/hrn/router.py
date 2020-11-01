import torch
from torch import nn
import torch.nn.functional as F
import pugh_torch as pt
from pugh_torch.modules import RandHashProj


class _NoEmptyUnits(Exception):
    """There are no empty units available"""


class HRN(nn.Module):
    """Routes data through a provided set of units"""

    def __init__(self, units, depth_thresh=1e-5, empty_thresh=0.2, expand_thresh=0.01):
        """
        Parameters
        ----------
        units : list
            List of ``HRNUnit``
        depth_thresh : float
            Stop routing once the produced residual is below this magnitude.
        empty_thresh : float
            While training, if there are empty units, randomly select one for
            routing if none of the populated units have a basis response
            above this threshold.
        expand_thresh : float
            While training, for a selected unit, if there are uninitialized
            vectors in its basis, then add a basis vector if the
            projection magnitude is below this threshold.
        """

        super().__init__()

        # Downstream logic relies on this
        # Its also just reasonable to try and initialize all units before
        # further populating a unit's basis.
        assert expand_thresh < empty_thresh

        self.units = nn.ModuleList(units)
        self.depth_thresh = depth_thresh
        self.empty_thresh = empty_thresh
        self.expand_thresh = expand_thresh

        # Make sure all units have the same hash size
        hash_size = units[0].hash_feat
        for i, unit in enumerate(units):
            assert (
                unit.hash_feat == hash_size
            ), f"Unit[{i}] hash hashing size {unit.hash_feat}; expected {hash_size}"

        # initial hasher (phi-0) for hashing the input vector.
        self.hasher = RandHashProj(hash_size)

    def __len__(self):
        return len(self.units)

    @property
    def hash_feat(self):
        return self.hasher.out_feat

    def forward_exemplar(self, x, depth=-1):
        """Forward pass of a single exemplar

        Parameters
        ----------
        x : torch.Tensor
            (c, h, w) input exemplar
        """

        if depth < 0:
            depth = 10000  # good enough

        # Add a singleton batch dimension to each exemplar
        x = x.unsqueeze(0)

        route = []
        # Will store the accumulated residuals
        output = torch.zeros(self.hash_feat)

        h0 = self.hasher(x.flatten(1))  # (b, out_feat)

        # Each element is ``(Unit, index)``
        available_units = dict(enumerate(self.units))

        def get_empty_units_idx():
            return [idx for idx, unit in available_units.items() if unit.is_empty]

        def get_non_empty_units_idx():
            return [idx for idx, unit in available_units.items() if not unit.is_empty]

        def init_random_empty_unit():
            empty_units_idx = get_empty_units_idx()

            if not empty_units_idx:
                raise _NoEmptyUnits

            # Select a random first unit
            idx = empty_units_idx[torch.randint(len(empty_units_idx), size=())]
            unit = available_units.pop(idx)
            route.append(idx)

            # Add the current hash to the unit's basis
            # h0 already has unit-norm
            unit.basis.insert(h.squeeze(0))

        h = h0
        for d in range(depth):
            non_empty_units_idx = get_non_empty_units_idx()
            if not non_empty_units_idx:
                # All units are empty, initialize a random empty unit.
                init_random_empty_unit()
                break

            # Choose a  unit based on basis response.
            projs = torch.stack(
                [available_units[idx].proj(h) for idx in non_empty_units_idx], dim=-1
            )
            mags = torch.linalg.norm(projs, dim=1)

            max_idx = int(torch.argmax(mags.squeeze()))
            max_mag = mags[max_idx]

            if max_mag < self.empty_thresh:
                # The best unit didn't respond very strongly.
                # Try and initialize an empty unit, if available.
                # Otherwise, continue with the routing algorithm
                try:
                    init_random_empty_unit()
                    break
                except _NoEmptyUnits:
                    pass

            idx = non_empty_units_idx[max_idx]
            unit = available_units.pop(idx)

            if max_mag < self.expand_thresh and not unit.is_full:
                unit.basis.insert(residual)
                # Recompute the projection with the updated basis
                proj = unit.proj(h)
            else:
                proj = projs[:, max_idx]

            residual = h - proj
            unit.select()

            output += residual.squeeze(0)
            route.append(idx)

            # Check other route-ending conditions
            if not available_units:
                # We have run out of available units
                break
            if d == depth - 1:
                # This is the last route
                break

            # Compute the next feature-map and hashed-feature-vector
            x, h = unit(x)

        return output, route

    def forward(self, x, depth=-1):
        """

        Parameters
        ----------
        x : list or torch.Tensor
            List of (c, H, W) Input data; or a single (b, c, H, W) tensor.
            Each input may have a different spatial shape.
            if provided as a single tensor, it will be unbinded into a list of
            tensors.
        depth : int
            Maximum number of routes to make.
            A negative value means to route until convergence (which is not
            guarenteed).

        Returns
        -------
        torch.Tensor
            (B, out_feat) Output feature vector
        list
            Routing path taken
        """

        if isinstance(x, torch.Tensor):
            x = torch.unbind(x)

        hashes, routes = [], []
        for exemplar in x:
            exemplar_hash, exemplar_route = self.forward_exemplar(exemplar, depth=depth)
            hashes.append(exemplar_hash)
            routes.append(exemplar_route)

        hashes = torch.stack(hashes, 0)
        return hashes, routes
