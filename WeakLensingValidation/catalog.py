# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Catalogue

from config_parser import ConfigParser

# This is used with the eval function
import dask.array as da
import dask.dataframe as df
import vaex
from vaex.convert import convert

import numpy as np

import os
import copy


class Catalog():
    """Catalog

    This class store a catalog as multiple DaskArrays.
    This class is instenciate from a config file.
    """

    def __init__(
        self,
        path=None,
        config=None,
        params=None,
    ):

        if isinstance(config, ConfigParser):
            self._config = config.config
        else:
            raise ValueError(
                f"config must be an instance of {ConfigParser}. "
                f"Got: {type(config)}"
            )

        self.read_catalog()

    def __getitem__(self, key):

        if not isinstance(key, str):
            raise KeyError("Key must be a string")
        if key not in self.column_names:
            raise KeyError(f"{key}")
        return self._columns[key]

    def read_catalog(self):

        raise NotImplementedError

    def _read_catalog(self, cat_config):
        """read catalog

        This method handle the reading of a catalog. It can read multiple
        catalogs and return a single Dataframe. If requested, it will keep
        track of the origin catalog for every objects.

        Parameters
        ----------
        cat_config : dict
            Configuration dictionnary
        """

        path_config = cat_config['path']

        # Here we handle the opeinig of multiple files like vaex.open_many()
        # but we also keep track of the catalog of origin for each objects if
        # requested.
        all_df = []
        for i, path in enumerate(path_config['path']):
            # First we check the extension of the file. If not ".hdf5" we
            # convert it. We cannot make memory mapped operations from ".fits"
            # file. The converted file is put in the workspace directory.
            ext = os.path.splitext(path)[1]
            file_name = os.path.split(path)[1]
            if ext != '.hdf5':
                new_path = \
                    self._config["workspace"]['path'] + '/' \
                    + file_name + '.hdf5'
                if ext != '.fits':
                    raise ValueError(f"Unreconized file format. Got: {ext}")
                if not os.path.exists(new_path):
                    convert(
                        path_input=path,
                        fs_options_input={},
                        fs_input=None,
                        path_output=new_path,
                        fs_options_output={},
                        fs_output=None,
                        progress=False,
                    )
            else:
                new_path = \
                    self._config["workspace"]['path'] + '/' + file_name
                if not os.path.exists(new_path):
                    os.symlink(path, new_path)

            # Now we handle the history if requested
            df_tmp = vaex.open(new_path)
            if path_config['keep_cat_history']:
                # Check if it has alreay been added
                if 'var_cat_history' not in df_tmp.column_names:
                    df_tmp['var_cat_history'] = \
                        np.ones(len(df_tmp), dtype=int) \
                        * path_config['var_cat_history'][i]
            all_df.append(df_tmp)
        self._df = vaex.concat(all_df)

    def _get_column(self, col_name, all_col_names):
        """get column

        Build all the column of the catalog and convert them to DaskArray

        Parameters
        ----------
        col_name : str, dict
            Column name in the original catalog or dictionnary with a
            description of how to build the column.
        all_col_names : list
            List of all the column names in the original catalog.

        Returns
        -------
        dask.array
            DaskArray of the column
        """

        if isinstance(col_name, str):
            return self._df[col_name].to_dask_array()
        elif isinstance(col_name, dict):
            cat_tmp = {}
            func = copy.copy(col_name['func'])
            for var_name in col_name['var']:
                cat_tmp[var_name] = self._df[var_name].to_dask_array()
                func = func.replace(
                    f"${var_name}$", f"cat_tmp['{var_name}']"
                )
            try:
                res = eval(func)
            except Exception as e:
                raise ValueError(
                        f"Error while evaluating function: \n{func}\n"
                        f"Got exception: \n{e}"
                    )
            return res


class GalaxyCatalog(Catalog):

    def __init__(
            self,
            config=None
    ):

        super().__init__(config=config)

    def read_catalog(self):
        """read catalogue

        This function is called during the initialization and build the galaxy
        catalog.
        """

        cat_config = self._config['galaxy_catalog']

        # First we read the catalog with vaex
        self._read_catalog(cat_config)

        # Now we set the column as dask arrays
        self.column_names = list(cat_config['columns'].keys())
        self._columns = {}
        for column_name in self.column_names:
            self._columns[column_name] = self._get_column(
                cat_config['columns'][column_name],
                self.column_names,
            )


class StarCatalog(Catalog):

    def __init__(
            self,
            config=None
    ):

        super().__init__(config=config)

    def read_catalog(self):
        """read catalogue

        This function is called during the initialization and build the star
        catalog.
        """

        cat_config = self._config['star_catalog']

        # First we read the catalog with vaex
        self._read_catalog(cat_config)

        # Now we set the column as dask arrays
        self.column_names = list(cat_config['columns'].keys())
        self._columns = {}
        for column_name in self.column_names:
            print(column_name)
            self._columns[column_name] = self._get_column(
                cat_config['columns'][column_name],
                self.column_names,
            )
