# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Original code by: Morgan Schmitz, Tobias Liaudat
# PSF residuals
# This code compute all sorts of residuals on the focal plane

import dask.array as da
# from dask.distributed import Client, LocalCluster

import numpy as np
import matplotlib.pyplot as plt

from catalog import StarCatalog
from config_parser import ConfigParser

import copy
from tqdm import tqdm
import os


def cantor_pairing_func(x, y):
    """cantor pairing function

    This function takes a couple of integer x, y and create a unique
    combination. See Wikipedia for more details:
    https://en.wikipedia.org/wiki/Pairing_function

    NOTE: f(x, y) != f(y, x)

    Parameters
    ----------
    x : int, array of int
        x value
    y : int, array of int
        y value

    Returns
    -------
    z : int, array of int
        Unique combination of (x, y)
    """
    z = 0.5 * (x + y) * (x + y + 1) + y
    return z.astype(int)


def inv_cantor_pairing_func(z):
    """inverse cantop pairing function

    This does the inverse operation of the Cantor pairing fucntion.
    See Wikipedia for more details:
    https://en.wikipedia.org/wiki/Pairing_function

    Parameters
    ----------
    z : int, array of int
        z unique value made from the cantor pairing function

    Returns
    -------
    x : int, array of int
        x value
    y : int, array of int
        y value
    """
    w = da.floor((da.sqrt(8.*z + 1) - 1)/2.)
    t = (w**2. + w)/2.
    y = z - t
    x = w - y

    return x.astype(int), y.astype(int)


class PSFResiduals():
    """PSFResiduals

    This class handle the plotting on the focal plane. It will bin the CCDs
    and compute, usually the mean, of a quantity in each of those bins. The
    user can also provide a custom evaluation function for more complex plots.
    The code can also performs basic operations between columns to compute
    residuals for example.

    Parameters
    ----------
    star_cat : StarCatalog
        Instance of a StarCatalog to use to make the plots
    config : ConfigParser
        Instance of the ConfigParser with all the informaton about the plots
    """

    def __init__(
        self,
        star_cat=None,
        config=None,
        # client=None,
    ):

        if not isinstance(star_cat, StarCatalog):
            raise ValueError(f"star_cat must be an instance of {StarCatalog}")
        self._starcat = star_cat
        if not isinstance(config, ConfigParser):
            raise ValueError(f"config must be an instance of {ConfigParser}")

        self.nbin_x = config.config['psf_residuals']['nbin_x']
        self.nbin_y = config.config['psf_residuals']['nbin_y']
        self._focal_plane_display = \
            config.config['psf_residuals']['focal_plane_display']
        self._npix_x = config.config['psf_residuals']['npix_x']
        self._npix_y = config.config['psf_residuals']['npix_y']
        self._n_ccd = config.config['psf_residuals']['n_ccd']
        self._plots = config.config['psf_residuals']['plot']

        self._initialize()
        self._set_workspace(config.config["workspace"])

    def process(self):
        """process

        Main method wich will first compute all the CCD bins and then make the
        plots.
        """

        # Compute all data to plot
        self._get_all_val_np()

        for plot_name in self._plots.keys():
            # Save the data for the plot
            self._save_results(plot_name)

            # Make the plot
            self._make_plot(plot_name)

    def _initialize(self):
        """initialization

        Here we set some parameters which are needed for the computation.

        NOTE: We neet to add here the handling of already existing runs.
        """

        self._nb_pixel = (self.nbin_x, self.nbin_y)

        self._grid = np.linspace(0, self._npix_x, self._nb_pixel[0] + 1), \
            np.linspace(0, self._npix_y, self._nb_pixel[1] + 1)

        ccd_ids = np.abs(np.unique(self._focal_plane_display))
        self._ccd_ids = ccd_ids[ccd_ids != 999]
        self._ccd_ids_2_ind = {
            ccd_nb: ind for ind, ccd_nb in enumerate(self._ccd_ids)
        }

    def _set_workspace(self, config_ws):
        """set_workspace

        Here we setup the workspace and output directory.

        Parameters
        ----------
        config_ws : dict
            Dictionnary with the workspace configuration
        """

        self._output_dir = os.path.join(config_ws['path'], 'psf_residuals')
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)

    def _get_all_val(self):
        """get_all_val

        WIP: Same as the numpy version below. But not used at the moment.
        """

        all_ccd_nb = self._starcat['ccd_nb']

        # From (x, y) we create a unique index
        # Get all unique indices on wich we iterate
        x_tmp = da.arange(1, self._nb_pixel[0]+1)
        y_tmp = da.arange(1, self._nb_pixel[1]+1)
        xx_tmp, yy_tmp = da.meshgrid(x_tmp, y_tmp)
        unique_xy = cantor_pairing_func(
            xx_tmp.ravel(),
            yy_tmp.ravel(),
        ).compute()

        plot_vals = {
            plot_name: self._get_plot_val(plot_name).persist()
            for plot_name in self._plots.keys()
        }

        ccd_maps = {plot_name: [] for plot_name in self._plots.keys()}
        for ccd_nb in tqdm(self._ccd_ids):
            mask_ccd = (all_ccd_nb == ccd_nb)
            xbins = da.digitize(self._starcat['x'][mask_ccd], self._grid[0])
            ybins = da.digitize(self._starcat['y'][mask_ccd], self._grid[1])
            xy_bins = cantor_pairing_func(xbins, ybins)
            xy_bins = xy_bins.persist()
            xy_bins.compute_chunk_sizes()
            mask_bins = [(xy_bins == n) for n in unique_xy]
            for plot_name in plot_vals.keys():
                plot_tmp = plot_vals[plot_name][mask_ccd].persist()
                plot_tmp.compute_chunk_sizes()
                plot = da.from_array([
                    da.mean(plot_tmp[mask_bin]).compute()
                    for mask_bin in mask_bins
                ])
                ccd_maps[plot_name].append(plot.reshape(
                    (self._nb_pixel[1], self._nb_pixel[0])
                ).compute())

    def _get_all_val_np(self):
        """get_all_val_np

        Get all the values for the different bins.

        NOTE: This is the numpy version which is faster than dask and do not
        use lot of memory.. At some point it could be nice to have a dask
        version of this function.
        """

        all_ccd_nb = self._starcat['ccd_nb'].compute()

        x = self._starcat['x'].compute()
        y = self._starcat['y'].compute()

        # From (x, y) we create a unique index
        # Get all unique indices on wich we iterate
        x_tmp = da.arange(1, self._nb_pixel[0]+1).compute()
        y_tmp = da.arange(1, self._nb_pixel[1]+1).compute()
        xx_tmp, yy_tmp = da.meshgrid(x_tmp, y_tmp)
        unique_xy = cantor_pairing_func(
            xx_tmp.ravel(),
            yy_tmp.ravel(),
        ).compute()

        # Get the columns to plot or operation between columns
        # Not optimal because here we load all the columns in memory
        plot_vals = {
            plot_name: self._get_plot_val(plot_name).compute()
            for plot_name in self._plots.keys()
        }

        # Now we loop over all CCDs.
        # Then we loop over each plots.
        # Finally we iterate over each bins.
        # Thanks to the unique indices we can avoid nested for loops and use
        # comprehension lists which are faster.
        self._ccd_maps = {plot_name: [] for plot_name in self._plots.keys()}
        for ccd_nb in tqdm(self._ccd_ids):
            mask_ccd = (all_ccd_nb == ccd_nb)
            xbins = np.digitize(x[mask_ccd], self._grid[0])
            ybins = np.digitize(y[mask_ccd], self._grid[1])
            xy_bins = cantor_pairing_func(xbins, ybins)
            mask_bins = [(xy_bins == n) for n in unique_xy]
            for plot_name in plot_vals.keys():

                plot_tmp = plot_vals[plot_name][mask_ccd]
                plot = np.array([
                    self._plots[plot_name]['eval_func'](plot_tmp[mask_bin])
                    for mask_bin in mask_bins
                ])
                self._ccd_maps[plot_name].append(
                    plot.reshape((self._nb_pixel[1], self._nb_pixel[0]))
                )

    def _save_results(self, plot_name):
        """save results

        Save the focal plane data for each plots as a numpy file.

        Parameters
        ----------
        plot_name : str
            Name of the plot to make among the one requested in the config
            file
        """

        output_name = os.path.join(self._output_dir, plot_name)
        output_name += '.npy'

        ccd_map = self._ccd_maps[plot_name]

        np.save(output_name, ccd_map)

    def _make_plot(self, plot_name):
        """make_plot

        Make one focal plot.

        Parameters
        ----------
        plot_name : str
            Name of the plot to make among the one requested in the config
            file
        """

        nrow, ncol = self._focal_plane_display.shape
        fig, axes = plt.subplots(
            nrows=nrow,
            ncols=ncol,
            figsize=(25, 12),
            dpi=400
        )

        output_name = os.path.join(self._output_dir, plot_name)
        output_name += '.png'

        vmax = np.nanmax(self._ccd_maps[plot_name])
        vmin = np.nanmin(self._ccd_maps[plot_name])

        # Remove un-used CCDs
        mask_ccd = np.where(self._focal_plane_display == 999)
        for ax in axes[mask_ccd]:
            ax.axis('off')

        # Loop over all CCDs
        for row in range(nrow):
            for col in range(ncol):
                ccd_nb = self._focal_plane_display[row, col]
                if ccd_nb == 999:
                    continue

                if ccd_nb < 0:
                    def flip_func(x): return np.flipud(x)
                    ccd_nb = abs(ccd_nb)
                else:
                    # The code return the CCDs upside down by default
                    flip_func = np.fliplr
                ccd_ind = self._ccd_ids_2_ind[ccd_nb]
                ax = axes[row, col]
                im = ax.imshow(
                    flip_func(self._ccd_maps[plot_name][ccd_ind]),
                    # np.flipud(self._ccd_maps[plot_name][ccd_nb].T),
                    interpolation='Nearest',
                    cmap='inferno',
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        plt.suptitle(self._plots[plot_name]['name'])
        plt.subplots_adjust(right=0.6)
        cbar_ax = fig.add_axes([0.65, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(output_name, bbox_inches='tight')
        plt.close()

    def _get_plot_val(self, plot_name):
        """get_plot_val

        Get the value to plot, either a column or an operation between
        columns.

        Parameters
        ----------
        plot_name : str
            Name of the plot to make among the one requested in the config
            file

        Returns
        -------
        res : da.array
            Dask array with the value to plot
        """

        plot = self._plots[plot_name]['plot']
        if isinstance(plot, str):
            return self._starcat[plot]
        if isinstance(plot, dict):
            cat_tmp = {}
            func = copy.deepcopy(plot['func'])
            for var_name in plot['var']:
                cat_tmp[var_name] = self._starcat[var_name]
                func = func.replace(
                    f"${var_name}$", f"cat_tmp['{var_name}']"
                )
            try:
                res = eval(func)
            except Exception as e:
                raise ValueError(
                    f"Error while evaluating function in plot_residuals:"
                    f"\n{func}\n"
                    f"Got exception: \n{e}"
                )
            return res
