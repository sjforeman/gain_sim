"""Core routines."""

import numpy as np

from caput import config, mpiutil, tools
from draco.core import task
from draco.util import random


class SimulateGainVariationStack(task.SingleTask, random.RandomTask):
    """Simulate a sidereal stack with frequency masks and gain variations.

    Attributes
    ----------
    rfi_mask_file : str
        Path to .npy file containing RFI masks, as single array with dimensions
        [nday, nfreq, nra].
    gain_amp_std : float
        Standard deviation of fractional gain variations.
    gain_phase_std : float
        Standard deviation of gain phase variations, in rad.
    """

    rfi_mask_file = config.Property(proptype=str)
    gain_amp_std = config.Property(proptype=float)
    gain_phase_std = config.Property(proptype=float)

    def setup(self, manager=None):
        """Load RFI masks and distribute to all ranks."""
        # Load RFI mask array on rank 0, and distribute shape to all ranks
        if self.comm.rank == 0:
            self.rfi_masks = np.load(self.rfi_mask_file)
            rfi_masks_shape = np.array(self.rfi_masks.shape)
        else:
            rfi_masks_shape = np.array([0, 0, 0])

        mpiutil.barrier()
        self.comm.Bcast(rfi_masks_shape, root=0)

        # Distribute RFI mask array to all ranks
        if self.comm.rank != 0:
            self.rfi_masks = np.zeros(rfi_masks_shape, dtype=bool)
        self.comm.Bcast(self.rfi_masks, root=0)

    def process(self, data):
        """Simulate a sidereal stack with random gain variations.

        For simplicity, the gain variations are assumed to be uncorrelated between
        frequencies, and times.

        Parameters
        ----------
        data : :class:`containers.SiderealStream`
            The input visibilities to be converted into a simulated stack.
            The visibilities are modified in-place.

        Returns
        -------
        data : same as parameter `data`
            The simulated stack.
        """
        data.redistribute("stack")

        ndays = self.rfi_masks.shape[0]
        nfreq = self.rfi_masks.shape[1]
        nra = self.rfi_masks.shape[2]

        # Check that frequency and RA axes match between input stream and RFI masks
        if data.vis.shape[0] != nfreq:
            raise ValueError(
                "Frequency axis mismatch between input stream and RFI masks: "
                f"{data.vis.shape[0]} vs. {nfreq}"
            )

        if data.vis.shape[2] != nra:
            raise ValueError(
                "RA axis mismatch between input stream and RFI masks: "
                f"{data.vis.shape[2]} vs. {nra}"
            )

        vis = data.vis[:]
        nstack = data.vis.local_shape[1]

        # Construct RFI-mask-weighted sum over gain variations.
        # Do so by looping over days, to save memory
        gain_prefactor = np.zeros((nfreq, nstack, nra), dtype=np.complex64)
        for i in range(ndays):
            self.log.debug(f"Computing gains for day {i}")
            # Each gain variation is A * exp(i * Phi), where A and Phi are Gaussian-
            # distributed
            gain_prefactor += (
                self.rfi_masks[i][:, np.newaxis, :]
                * self.rng.standard_normal(size=(nfreq, nstack, nra))
                * self.gain_amp_std
                * np.exp(
                    1j
                    * self.rng.standard_normal(size=(nfreq, nstack, nra))
                    * self.gain_amp_std
                )
            )

        # Normalize gain sum by sum of RFI masks over days
        gain_prefactor *= tools.invert_no_zero(
            np.sum(self.rfi_masks, axis=0)[:, np.newaxis, :]
        )

        # Multiply visibilities by RFI-mask-weighted gains
        vis *= 1 + gain_prefactor

        return data
