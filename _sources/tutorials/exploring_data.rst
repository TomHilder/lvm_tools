Tutorial: Exploring LVM Data
============================

This tutorial demonstrates how to explore and visualise LVM data, including spatial distributions and spectral properties.

Prerequisites
-------------

This tutorial assumes you have completed :doc:`loading_data` and have access to LVM DRP files. We will use matplotlib for visualisation:

.. code-block:: bash

   pip install matplotlib

Loading the Data
----------------

Begin by loading a tile:

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import matplotlib.pyplot as plt
   from lvm_tools import LVMTile

   # Load a tile
   tile = LVMTile.from_file(Path("/path/to/lvm-drp-observation.fits"))

   # Extract the dataset for convenience
   ds = tile.data

   print(tile)

Plotting Sky Positions
----------------------

Visualise the spatial distribution of fibres on the sky:

.. code-block:: python

   # Extract coordinates
   ra = ds["ra"].values.flatten()
   dec = ds["dec"].values.flatten()

   # Create a scatter plot of fibre positions
   fig, ax = plt.subplots(figsize=(8, 8))

   scatter = ax.scatter(ra, dec, c=np.arange(len(ra)), cmap="viridis", s=10)
   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title("LVM Fibre Positions")
   ax.set_aspect("equal")

   # Invert RA axis (standard astronomical convention)
   ax.invert_xaxis()

   plt.colorbar(scatter, ax=ax, label="Fibre index")
   plt.tight_layout()
   plt.show()

Colour-Coding by IFU Bundle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LVM uses three science IFU bundles. Visualise their spatial arrangement:

.. code-block:: python

   # Get IFU labels
   ifu_labels = ds["ifu_label"].values.flatten()
   unique_ifus = np.unique(ifu_labels)

   fig, ax = plt.subplots(figsize=(8, 8))

   colors = {"Sci1": "C0", "Sci2": "C1", "Sci3": "C2"}
   for ifu in unique_ifus:
       mask = ifu_labels == ifu
       ax.scatter(ra[mask], dec[mask], c=colors.get(ifu, "gray"),
                  label=ifu, s=15, alpha=0.7)

   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title("LVM IFU Bundle Positions")
   ax.set_aspect("equal")
   ax.invert_xaxis()
   ax.legend()
   plt.tight_layout()
   plt.show()

Colour-Coding by Fibre Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify fibres with different quality flags:

.. code-block:: python

   # Get fibre status
   fibre_status = ds["fibre_status"].values.flatten()
   unique_status = np.unique(fibre_status)

   fig, ax = plt.subplots(figsize=(8, 8))

   status_colors = {0: "green", 1: "orange", 2: "red", 3: "gray"}
   status_labels = {0: "Good", 1: "Warning", 2: "Bad", 3: "Unknown"}

   for status in unique_status:
       mask = fibre_status == status
       ax.scatter(ra[mask], dec[mask],
                  c=status_colors.get(status, "black"),
                  label=f"Status {status} ({status_labels.get(status, 'Unknown')})",
                  s=15, alpha=0.7)

   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title("Fibre Status Distribution")
   ax.set_aspect("equal")
   ax.invert_xaxis()
   ax.legend()
   plt.tight_layout()
   plt.show()

Plotting Individual Spectra
---------------------------

Extract and plot spectra from individual spaxels:

.. code-block:: python

   # Get wavelength grid
   wavelength = ds["wavelength"].values

   # Select a spaxel index
   spaxel_idx = 100

   # Extract flux and inverse variance (compute from Dask)
   flux = ds["flux"].isel(tile=0, spaxel=spaxel_idx).values
   ivar = ds["i_var"].isel(tile=0, spaxel=spaxel_idx).values

   # Compute uncertainty (standard deviation)
   sigma = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)

   # Get position of this spaxel
   spaxel_ra = float(ds["ra"].isel(tile=0, spaxel=spaxel_idx))
   spaxel_dec = float(ds["dec"].isel(tile=0, spaxel=spaxel_idx))

   # Plot
   fig, ax = plt.subplots(figsize=(12, 4))

   ax.plot(wavelength, flux, "k-", lw=0.5, label="Flux")
   ax.fill_between(wavelength, flux - sigma, flux + sigma,
                   alpha=0.3, color="C0", label=r"$\pm 1\sigma$")

   ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
   ax.set_ylabel(r"Flux (erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$)")
   ax.set_title(f"Spectrum at RA={spaxel_ra:.4f}, Dec={spaxel_dec:.4f}")
   ax.legend()
   ax.set_xlim(wavelength.min(), wavelength.max())
   plt.tight_layout()
   plt.show()

Zooming into Spectral Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examine specific wavelength regions:

.. code-block:: python

   # Define wavelength regions of interest
   regions = {
       r"H$\beta$ + [OIII]": (4800, 5100),
       r"H$\alpha$ + [NII]": (6500, 6650),
       r"[SII] doublet": (6700, 6750),
   }

   fig, axes = plt.subplots(1, 3, figsize=(14, 4))

   for ax, (name, (wl_min, wl_max)) in zip(axes, regions.items()):
       # Select wavelength range
       mask = (wavelength >= wl_min) & (wavelength <= wl_max)
       wl_subset = wavelength[mask]
       flux_subset = flux[mask]
       sigma_subset = sigma[mask]

       ax.plot(wl_subset, flux_subset, "k-", lw=0.8)
       ax.fill_between(wl_subset,
                       flux_subset - sigma_subset,
                       flux_subset + sigma_subset,
                       alpha=0.3, color="C0")

       ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
       ax.set_ylabel(r"Flux")
       ax.set_title(name)

   plt.tight_layout()
   plt.show()

Plotting Multiple Spectra
-------------------------

Compare spectra from different spatial locations:

.. code-block:: python

   # Select multiple spaxels
   spaxel_indices = [50, 100, 150, 200, 250]

   fig, ax = plt.subplots(figsize=(12, 6))

   for i, idx in enumerate(spaxel_indices):
       flux_i = ds["flux"].isel(tile=0, spaxel=idx).values
       ra_i = float(ds["ra"].isel(tile=0, spaxel=idx))
       dec_i = float(ds["dec"].isel(tile=0, spaxel=idx))

       # Offset for visibility
       offset = i * 1e-15
       ax.plot(wavelength, flux_i + offset, lw=0.5,
               label=f"({ra_i:.3f}, {dec_i:.3f})")

   ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
   ax.set_ylabel(r"Flux + offset")
   ax.set_title("Spectra from Multiple Positions")
   ax.legend(title="(RA, Dec)", fontsize=8)
   ax.set_xlim(wavelength.min(), wavelength.max())
   plt.tight_layout()
   plt.show()

Spatial Maps of Spectral Properties
-----------------------------------

Create maps of integrated flux in specific wavelength windows:

.. code-block:: python

   # Define a wavelength window around H-alpha
   ha_min, ha_max = 6555.0, 6575.0
   wl_mask = (wavelength >= ha_min) & (wavelength <= ha_max)

   # Compute integrated flux for all spaxels
   # Note: this triggers Dask computation
   flux_cube = ds["flux"].isel(tile=0).values  # Shape: (n_spaxels, n_wavelengths)
   ha_flux = np.nansum(flux_cube[:, wl_mask], axis=1)

   # Plot spatial map
   fig, ax = plt.subplots(figsize=(8, 8))

   scatter = ax.scatter(ra, dec, c=ha_flux, cmap="plasma", s=20,
                        vmin=np.nanpercentile(ha_flux, 5),
                        vmax=np.nanpercentile(ha_flux, 95))

   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title(r"H$\alpha$ Integrated Flux Map")
   ax.set_aspect("equal")
   ax.invert_xaxis()

   plt.colorbar(scatter, ax=ax, label=r"Flux (erg cm$^{-2}$ s$^{-1}$)")
   plt.tight_layout()
   plt.show()

Continuum-Subtracted Maps
~~~~~~~~~~~~~~~~~~~~~~~~~

Create maps with local continuum subtraction:

.. code-block:: python

   # Define continuum windows on either side of H-alpha
   cont_blue = (6500.0, 6540.0)
   cont_red = (6590.0, 6630.0)

   # Masks for each window
   blue_mask = (wavelength >= cont_blue[0]) & (wavelength <= cont_blue[1])
   red_mask = (wavelength >= cont_red[0]) & (wavelength <= cont_red[1])

   # Compute median continuum
   cont_blue_flux = np.nanmedian(flux_cube[:, blue_mask], axis=1)
   cont_red_flux = np.nanmedian(flux_cube[:, red_mask], axis=1)
   continuum = 0.5 * (cont_blue_flux + cont_red_flux)

   # Subtract continuum from H-alpha flux
   ha_flux_cont_sub = ha_flux - continuum * np.sum(wl_mask)

   # Plot
   fig, axes = plt.subplots(1, 2, figsize=(14, 6))

   # Raw H-alpha flux
   sc1 = axes[0].scatter(ra, dec, c=ha_flux, cmap="plasma", s=20,
                         vmin=np.nanpercentile(ha_flux, 5),
                         vmax=np.nanpercentile(ha_flux, 95))
   axes[0].set_title(r"H$\alpha$ (raw)")
   plt.colorbar(sc1, ax=axes[0])

   # Continuum-subtracted
   sc2 = axes[1].scatter(ra, dec, c=ha_flux_cont_sub, cmap="RdBu_r", s=20,
                         vmin=np.nanpercentile(ha_flux_cont_sub, 5),
                         vmax=np.nanpercentile(ha_flux_cont_sub, 95))
   axes[1].set_title(r"H$\alpha$ (continuum subtracted)")
   plt.colorbar(sc2, ax=axes[1])

   for ax in axes:
       ax.set_xlabel("Right Ascension (deg)")
       ax.set_ylabel("Declination (deg)")
       ax.set_aspect("equal")
       ax.invert_xaxis()

   plt.tight_layout()
   plt.show()

Signal-to-Noise Maps
--------------------

Visualise data quality across the field:

.. code-block:: python

   # Compute median S/N per spaxel
   ivar_cube = ds["i_var"].isel(tile=0).values
   sigma_cube = np.where(ivar_cube > 0, 1.0 / np.sqrt(ivar_cube), np.nan)
   snr_cube = flux_cube / sigma_cube

   # Median S/N across wavelength
   median_snr = np.nanmedian(snr_cube, axis=1)

   fig, ax = plt.subplots(figsize=(8, 8))

   scatter = ax.scatter(ra, dec, c=median_snr, cmap="viridis", s=20,
                        vmin=0, vmax=np.nanpercentile(median_snr, 95))

   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title("Median Signal-to-Noise Ratio")
   ax.set_aspect("equal")
   ax.invert_xaxis()

   plt.colorbar(scatter, ax=ax, label="Median S/N per pixel")
   plt.tight_layout()
   plt.show()

Line Spread Function Visualisation
----------------------------------

Examine the spectral resolution across the field:

.. code-block:: python

   # Get LSF sigma (already converted from FWHM)
   lsf_cube = ds["lsf_sigma"].isel(tile=0).values

   # Median LSF at a reference wavelength (e.g., H-alpha)
   ha_wl_idx = np.argmin(np.abs(wavelength - 6563.0))
   lsf_at_ha = lsf_cube[:, ha_wl_idx]

   # Convert sigma to FWHM for display
   SIGMA_TO_FWHM = 2.355
   lsf_fwhm = lsf_at_ha * SIGMA_TO_FWHM

   fig, ax = plt.subplots(figsize=(8, 8))

   scatter = ax.scatter(ra, dec, c=lsf_fwhm, cmap="coolwarm", s=20)

   ax.set_xlabel("Right Ascension (deg)")
   ax.set_ylabel("Declination (deg)")
   ax.set_title(r"LSF FWHM at H$\alpha$ ($\mathrm{\AA}$)")
   ax.set_aspect("equal")
   ax.invert_xaxis()

   plt.colorbar(scatter, ax=ax, label=r"FWHM ($\mathrm{\AA}$)")
   plt.tight_layout()
   plt.show()

Summary Statistics
------------------

Generate summary statistics for the observation:

.. code-block:: python

   # Compute various statistics
   n_spaxels = len(ra)
   n_wavelengths = len(wavelength)
   n_good_fibres = np.sum(fibre_status == 0)

   # Wavelength coverage
   wl_min, wl_max = wavelength.min(), wavelength.max()
   wl_step = np.median(np.diff(wavelength))

   # Spatial coverage
   ra_range = ra.max() - ra.min()
   dec_range = dec.max() - dec.min()

   # Data quality
   valid_fraction = np.mean(~np.isnan(flux_cube))
   median_snr_all = np.nanmedian(snr_cube)

   print("=" * 50)
   print("LVM Observation Summary")
   print("=" * 50)
   print(f"Tile ID:           {tile.meta.tile_id}")
   print(f"Exposure:          {tile.meta.exp_num}")
   print(f"DRP Version:       {tile.meta.drp_ver}")
   print("-" * 50)
   print(f"Spaxels:           {n_spaxels}")
   print(f"Good fibres:       {n_good_fibres} ({100*n_good_fibres/n_spaxels:.1f}%)")
   print(f"Wavelength points: {n_wavelengths}")
   print(f"Wavelength range:  {wl_min:.1f} - {wl_max:.1f} Å")
   print(f"Wavelength step:   {wl_step:.3f} Å")
   print("-" * 50)
   print(f"RA range:          {ra.min():.4f} - {ra.max():.4f} deg ({ra_range*60:.2f} arcmin)")
   print(f"Dec range:         {dec.min():.4f} - {dec.max():.4f} deg ({dec_range*60:.2f} arcmin)")
   print("-" * 50)
   print(f"Valid data:        {100*valid_fraction:.1f}%")
   print(f"Median S/N:        {median_snr_all:.1f}")
   print("=" * 50)

Next Steps
----------

- :doc:`data_configuration`: Learn to configure data processing
- :doc:`fitting_workflow`: Apply configurations in a fitting pipeline
