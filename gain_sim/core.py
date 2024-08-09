"""Core routines."""

import numpy as np

from caput import config, mpiutil, tools
from draco.core import task
from draco.util import random
from draco.util.exception import ConfigError


class SimulateGainVariationStack(task.SingleTask, random.RandomTask):
    """Simulate a sidereal stack with frequency masks and gain variations.

    Attributes
    ----------
    rfi_mask_file : str
        Path to .npy file containing RFI masks, as single array with dimensions
        [nday, nfreq, nra].
    mode : str
        Type of gain variantions to simulate:
        - "random": Gaussian amplitudes and phases, uncorrelated in frequency,
        RA, and baseline+pol
        - "per_baseline": one Gaussian amplitude per day/baseline+pol, constant
        in frequency and RA; one Gaussian delay per day/baseline+pol, constant
        in frequency and RA
        - "per_day": one Gaussian amplitude per day, constant in frequency,
        RA, and baseline+pol; one Gaussian delay per day, constant in frequency,
        RA, and baseline+pol
    gain_amp_std : float
        Standard deviation of fractional gain variations.
    gain_phase_std : float
        Standard deviation of gain phase variations, in rad. Ignored if mode
        is "constant" or "per_baseline".
    gain_delay_std : float
        Standard deviation of gain delay variations, in ps. Ignored if mode
        is "random".
    """

    rfi_mask_file = config.Property(proptype=str)
    mode = config.enum(["random", "per_baseline", "per_day"])
    gain_amp_std = config.Property(proptype=float)
    gain_phase_std = config.Property(proptype=float)
    gain_delay_std = config.Property(proptype=float)

    def setup(self, manager=None):
        """Load RFI masks and distribute to all ranks."""
        # Check for valid mode
        if self.mode not in ["random", "per_baseline", "per_day"]:
            raise ConfigError(f"Invalid mode specified: {self.mode}")

        # Check for gain sim parameters
        if self.gain_amp_std is None:
            raise ConfigError("Must specify gain_amp_std")

        if self.mode == "random" and self.gain_phase_std is None:
            raise ConfigError("Must specify gain_phase_std when mode=random")
        elif self.mode in ["per_baseline", "per_day"] and self.gain_delay_std is None:
            raise ConfigError("Must specify gain_delay_std when mode!=random")

        # Load RFI mask array on rank 0, and distribute shape to all ranks
        if self.comm.rank == 0:
            self.rfi_masks = np.load(self.rfi_mask_file)
            rfi_masks_shape = np.array(self.rfi_masks.shape)
        else:
            rfi_masks_shape = np.array([0, 0, 0])

        self.comm.barrier()
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

        vis = data.vis[:].local_array
        nstack = data.vis.local_shape[1]

        # Construct RFI-mask-weighted sum over gain variations.
        # Do so by looping over days, to save memory
        if self.mode == "random":
            # Generate gains with random amplitudes and phases that are uncorrelated
            # in day, frequency, baseline+pol, and RA
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
                        * self.gain_phase_std
                    )
                )

        elif self.mode == "per_baseline":
            # Generate gains with random amplitudes and delays that are uncorrelated
            # in day and baseline+pol but constant in frequency and RA
            gain_prefactor = np.zeros((nfreq, nstack, nra), dtype=np.complex64)
            for i in range(ndays):
                self.log.debug(f"Computing gains for day {i}")
                # Each gain variation is A * exp(2 pi i nu dtau),
                # where A and dtau are Gaussian-distributed
                gain_prefactor += (
                    self.rfi_masks[i][:, np.newaxis, :]
                    * self.rng.standard_normal(size=nstack)[np.newaxis, :, np.newaxis]
                    * self.gain_amp_std
                    * np.exp(
                        1j
                        * 2
                        * np.pi
                        * data.freq[:, np.newaxis, np.newaxis]
                        * 1e6
                        * self.rng.standard_normal(size=nstack)[
                            np.newaxis, :, np.newaxis
                        ]
                        * self.gain_delay_std
                        * 1e-12
                    )
                )

        elif self.mode == "per_day":
            # Generate gains with random amplitudes and delays that are uncorrelated
            # in day but constant in frequency, baseline+pol, and RA

            # Initialize RNG (must be done on all ranks)
            _ = self.rng

            if self.comm.rank == 0:
                # Generate gain array with shape (ndays, nfreq)
                gains = (
                    self.rng.standard_normal(size=ndays)[:, np.newaxis]
                    * self.gain_amp_std
                    * np.exp(
                        1j
                        * 2
                        * np.pi
                        * data.freq[np.newaxis, :]
                        * 1e6
                        * self.rng.standard_normal(size=ndays)[:, np.newaxis]
                        * self.gain_delay_std
                        * 1e-12
                    )
                ).astype(np.complex64)
            else:
                gains = np.zeros((ndays, nfreq)).astype(np.complex64)

            self.comm.barrier()
            self.comm.Bcast(gains, root=0)

            # Compute gain-weighted RFI mask sum, with shape (nfreq, nra)
            gain_prefactor = np.sum(gains[:, :, np.newaxis] * self.rfi_masks, axis=0)

        if self.mode in ["random", "per_baseline"]:
            # Normalize gain sum by sum of RFI masks over days.
            # gain_prefactor has shape (nfreq, nstack, nra)
            gain_prefactor *= tools.invert_no_zero(
                np.sum(self.rfi_masks, axis=0)[:, np.newaxis, :]
            )

            # Multiply visibilities by RFI-mask-weighted gains
            vis *= 1 + gain_prefactor

        elif self.mode == "per_day":
            # Normalize gain sum by sum of RFI masks over days.
            # gain_prefactor has shape (nfreq, nra)
            gain_prefactor *= tools.invert_no_zero(np.sum(self.rfi_masks, axis=0))
            # Multiply visibilities by RFI-mask-weighted gains
            vis *= 1 + gain_prefactor[:, np.newaxis, :]

        return data
