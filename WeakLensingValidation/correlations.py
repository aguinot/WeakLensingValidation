# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Correlations
# This code handles the call to TreeCorr to compute the correlation functions

from config_parser import ConfigParser

import treecorr
import numpy as np
from math import ceil

import os
import shutil
import copy
import gc


_default_tc_config = {
    'ra_units': 'degrees',
    'dec_units': 'degrees',
    'min_sep': 2,
    'max_sep': 250,
    'sep_units': 'arcmin',
    'nbins': 32,
    'bin_slop': None,
    'npatch': 50,
    'cov_type': 'jackknife',
    'low_mem': False,
}

_max_size_low_mem = 1_000_000


class Correlations():
    """Correlations

        Parameters
        ----------
        config_corr : dict,
            Dictionnary with the parameters on how to compute the correlations
        config : _type_, optional
            Instance of the ConfigParser. This is used to setup the working
            directory.
        """

    def __init__(
        self,
        config_corr,
        config,
    ):

        if not isinstance(config, ConfigParser):
            raise ValueError(f"config must be an instance of {ConfigParser}")

        if not isinstance(config_corr, dict):
            raise ValueError("config_corr must be a dictionnary")

        self._set_workspace(config.config["workspace"])
        self._set_config(config_corr)

    def _set_workspace(self, config_ws):
        """set_workspace

        Here we setup the workspace and temporary directory used for the
        patches in low_mem mode which is the default.

        Parameters
        ----------
        config_ws : dict
            Dictionnary with the workspace configuration
        """

        self._tmp_dir = os.path.join(config_ws['path'], 'tmp_corr')
        if not os.path.exists(self._tmp_dir):
            os.mkdir(self._tmp_dir)
        else:
            # If the directory is not empty we remove it and create it again
            # It is necessary to have an empty dir because it can messed up
            # the computation if patches are used.
            if not len(os.listdir(self._tmp_dir)) == 0:
                try:
                    shutil.rmtree(self._tmp_dir)
                except Exception as e:
                    raise OSError(
                        "Error while removing already existing temporary "
                        f"directory at: {self._tmp_dir}. Got the following "
                        f"error:\n{e}"
                    )
                os.mkdir(self._tmp_dir)

    def _set_config(self, config_corr):
        """set_config

        Here we setup the configuration for TreeCorr. If a parameter is not
        provided we use default values.

        Defaults:
            ra_units: degrees
            dec_units: degrees
            min_sep: 2
            max_sep: 250
            sep_units: arcmin
            nbins: 32
            bin_slop: None
            npatch: 50
            cov_type: jackknife
            low_mem: False

        Parameters
        ----------
        config_corr : dict
            Dictionnary with the parameters on how to compute the correlations
        """

        config_tmp = copy.deepcopy(_default_tc_config)
        config_tmp.update(config_corr)

        self._tc_config = {
            'ra_units': config_tmp['ra_units'],
            'dec_units': config_tmp['dec_units'],
            'min_sep': config_tmp['min_sep'],
            'max_sep': config_tmp['max_sep'],
            'sep_units': config_tmp['sep_units'],
            'nbins': config_tmp['nbins'],
            'bin_slop': config_tmp['bin_slop'],
            'var_method': config_tmp['cov_type'],
        }

        print("config:\n", self._tc_config)

        self._npatch = config_tmp['npatch']
        self._low_mem = config_tmp['low_mem']

        # Maybe move this somewhere else
        self._cat = {}
        self._path_cen_patch = None
        self._patch_dir = None

    def process_shear_shear(
        self,
        ra,
        dec,
        g1,
        g2,
        w=None,
        ra_2=None,
        dec_2=None,
        g1_2=None,
        g2_2=None,
        w_2=None,
        reuse_center=False,
        output_name="corr_results.fits",
        save_cov=True,
    ):
        """process_shear_shear

        This is the main function users are going to call to compute the
        Shear-Shear correlations. The entries are provided as dask.Arrays.
        Once the Correlation class is intenciated, this method can be called
        multiple times. For example, if one wants to compute all the rho
        statistics using patches, it will call this function the first time
        with ``reuse_center=False`` and can switch to ``True`` on the folowing
        calls to avoid re-computing the patch centers. The results are saved
        and then the memory is freed. If no secondary catalog are provided
        the code performs auto-correlation.

        Parameters
        ----------
        ra : dask.array.core.Array
            Right ascension for catalog 1
        dec : dask.array.core.Array
            Declination for catalog 1
        g1 : dask.array.core.Array
            First component of the shear for catalog 1
        g2 : dask.array.core.Array
            Second component of the shear for catalog 1
        w : dask.array.core.Array, optional
            Weights for catalog 1, by default None
        ra_2 : dask.array.core.Array, optional
            Right ascension for catalog 2, by default None
        dec_2 : dask.array.core.Array, optional
            Declination for catalog 2, by default None
        g1_2 : dask.array.core.Array, optional
            First component of the shear for catalog 2, by default None
        g2_2 : dask.array.core.Array, optional
            Second component of the shear for catalog 2, by default None
        w_2 : dask.array.core.Array, optional
            Weights for catalog 2, by default None
        reuse_center : bool, optional
            On the second call of this method this parameter can be set to True
            to avoid re-compting the centers of the patches, by default False
        output_name : str, optional
            Path of the output file with the correlation, by default
            "corr_results.fits"
        save_cov: bool, optional
            Weither to save the covariance matrix, by default True
        """

        # Load data
        print("Data")
        self._set_data(ra, dec, g1, g2, w, ra_2, dec_2, g1_2, g2_2, w_2)

        # Do patches
        if self._npatch != 0:
            print("Make patches")
            self._make_patch_dir()
            if reuse_center:
                if isinstance(self._path_cen_patch, type(None)):
                    raise ValueError(
                        "Patch centers file do not exist. It has to be created"
                        " on first function call by setting: "
                        "reuse_center=False"
                    )
            else:
                self._make_patch_cen()

        # Make treecorr Catalogs
        print("TC cat0")
        tc_cat0 = treecorr.Catalog(
            ra=self._cat[0]['ra'],
            dec=self._cat[0]['dec'],
            g1=self._cat[0]['g1'],
            g2=self._cat[0]['g2'],
            w=self._cat[0]['w'],
            ra_units='degrees',
            dec_units='degrees',
            patch_centers=self._path_cen_patch,
            save_patch_dir=self._patch_dir,
        )
        print("TC cat1")
        tc_cat1 = treecorr.Catalog(
            ra=self._cat[1]['ra'],
            dec=self._cat[1]['dec'],
            g1=self._cat[1]['g1'],
            g2=self._cat[1]['g2'],
            w=self._cat[1]['w'],
            ra_units='degrees',
            dec_units='degrees',
            patch_centers=self._path_cen_patch,
            save_patch_dir=self._patch_dir,
        )

        # Init the shear-shear correlation class
        print("Init GG")
        self.GG = treecorr.GGCorrelation(self._tc_config)

        # Run correlation
        print("Run GG...")
        self.GG.process(
            tc_cat0,
            tc_cat1,
            low_mem=self._low_mem
        )

        # Save the results to file
        self._save_results(output_name, save_cov)

        # Free the memory
        self.GG.clear()
        tc_cat0.clear_cache()
        tc_cat1.clear_cache()
        del self.GG
        del tc_cat0
        del tc_cat1
        gc.collect()

    def _save_results(self, output_name, save_cov=True):
        """save_results

        Here we save the results to file.

        Parameters
        ----------
        output_name : str
            Path of the output file with the correlation
        save_cov : bool, optional
            Weither to save the covariance matrix, by default True
        """

        self.GG.write(output_name)

        if save_cov:
            cov_output_name = os.path.splitext(output_name)[0]
            cov_output_name += '_cov.npy'
            np.save(cov_output_name, self.GG.cov)

    def _set_data(
            self,
            ra, dec, g1, g2, w,
            ra_2, dec_2, g1_2, g2_2, w_2,
    ):
        """set_data

        Here we prepare the data and handle the auto-correlation case.

        Parameters
        ----------
        ra : dask.array.core.Array
            Right ascension for catalog 1
        dec : dask.array.core.Array
            Declination for catalog 1
        g1 : dask.array.core.Array
            First component of the shear for catalog 1
        g2 : dask.array.core.Array
            Second component of the shear for catalog 1
        w : dask.array.core.Array, optional
            Weights for catalog 1, by default None
        ra_2 : dask.array.core.Array, optional
            Right ascension for catalog 2, by default None
        dec_2 : dask.array.core.Array, optional
            Declination for catalog 2, by default None
        g1_2 : dask.array.core.Array, optional
            First component of the shear for catalog 2, by default None
        g2_2 : dask.array.core.Array, optional
            Second component of the shear for catalog 2, by default None
        w_2 : dask.array.core.Array, optional
            Weights for catalog 2, by default None
        """

        self._set_cat(ra, dec, g1, g2, w, 0)

        # We check for catalog 2
        check_cat2 = [
            isinstance(ra_2, type(None)),
            isinstance(dec_2, type(None)),
            isinstance(g1_2, type(None)),
            isinstance(g2_2, type(None))
        ]
        if all(check_cat2):
            # No cat 2
            self._set_cat(ra, dec, g1, g2, w, 1)
        elif any(check_cat2):
            # Some cat 2
            raise ValueError(
                "Missing value for catalog 2. At least: [ra, dec, g1, g2] "
                "must be provided."
            )
        else:
            # All cat 2
            self._set_cat(ra_2, dec_2, g1_2, g2_2, w_2, 1)

    def _set_cat(self, ra, dec, g1, g2, w, num):
        """set_cat

        Here we load the data in memory before calling TreeCorr.

        Parameters
        ----------
        ra : dask.array.core.Array
            Right ascension for catalog 1
        dec : dask.array.core.Array
            Declination for catalog 1
        g1 : dask.array.core.Array
            First component of the shear for catalog 1
        g2 : dask.array.core.Array
            Second component of the shear for catalog 1
        w : dask.array.core.Array, optional
            Weights for catalog 1, by default None
        num : int
            ID of the catalog in [0, 1]
        """

        self._cat[num] = {
            'ra': ra.compute(),
            'dec': dec.compute(),
            'g1': g1.compute(),
            'g2': g2.compute(),
        }

        if isinstance(w, type(None)):
            self._cat[num]['w'] = None
        else:
            self._cat[num]['w'] = w.compute()

    def _make_patch_dir(self):
        """make_patch_dir

        We set the patch directory used in the low_mem case.
        """

        # We also setup the directory where the patch are stored
        self._patch_dir = os.path.join(self._tmp_dir, 'patches')
        if os.path.exists(self._patch_dir):
            shutil.rmtree(self._tmp_dir)
        else:
            os.mkdir(self._patch_dir)

    def _make_patch_cen(self):
        """make_patch_cen

        We create the patch center file used by the computation later on. This
        is done with a small part of the catalog to make it faster.
        """

        self._path_cen_patch = os.path.join(self._tmp_dir, 'patch_cen.dat')
        every_nth = 1
        cat_size = len(self._cat[0]['ra'])
        if cat_size > _max_size_low_mem:
            every_nth = int(ceil(cat_size/_max_size_low_mem))

        tc_cat = treecorr.Catalog(
            ra=self._cat[0]['ra'],
            dec=self._cat[0]['dec'],
            g1=self._cat[0]['g1'],
            g2=self._cat[0]['g2'],
            w=self._cat[0]['w'],
            ra_units='degrees',
            dec_units='degrees',
            every_nth=every_nth,
            npatch=self._npatch,
        )

        tc_cat.write_patch_centers(self._path_cen_patch)
        tc_cat.clear_cache()
        del tc_cat
        gc.collect()
