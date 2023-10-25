# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Abstract Catalogue

import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np  # noqa
import dask.array as da # noqa
import vaex

from wlv.config_parser import ConfigParser


class Catalog(ABC):
    """Stores a catalog.

    Do not use this class directly, use Star_catalog or Galaxy_catalog instead
    """

    def __init__(self, config=None):
        if isinstance(config, ConfigParser):
            self._config = config.config
        else:
            raise ValueError(
                f"config must be an instance of {ConfigParser}. "
                f"Got: {type(config)}"
            )
        self.column_names = None
        self._columns = None
        self._df = None
        self.read_catalog()  # contains all catalogs

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError("Key must be a string")
        if key not in self.column_names:
            raise KeyError(f"{key}")
        return self._columns[key]

    @abstractmethod
    def read_catalog(self):
        pass

    def _read_catalog(self, cat_config):
        """Read catalogs

        This method reads one or many catalog and returns a single Dataframe. If requested, the source file of each
        object will be recorded.

        Parameters
        ----------
        cat_config : dict
            Configuration dictionary
        Returns
        -------
        dataframe :
            dataframe of the catalog
        """

        path_config = cat_config["path"]
        catalog_files = self._convert_catalog_to_hdf5(
            path_config, self._config["workspace"]["path"]
        )

        if not path_config["keep_cat_history"]:
            return vaex.open_many(catalog_files)  # TODO check
        else:
            concatenated_df = None
            for i, catalog_file in enumerate(catalog_files):
                df = vaex.open(catalog_file)
                if "var_cat_history" not in df.column_names:
                    df["var_cat_history"] = vaex.vconstant(
                        path_config["var_cat_history"][i],
                        length=df.shape[0],
                        dtype=int,
                    )
                if concatenated_df is None:
                    concatenated_df = df
                else:
                    concatenated_df = concatenated_df.concat(df)
                print(
                    f"Done reading star catalog ...{catalog_file[-42:]}: {df.shape[0],} rows."
                )
        return concatenated_df

    def get_df(self):
        return self._df

    def get_columns(self):
        return self._columns

    @staticmethod
    def _convert_catalog_to_hdf5(list_of_catalog, workspace):
        """Converts a list of catalog in fits or hdf5 format to parquet.

        If the format is already parquet but not in the current workspace, then a symlink will be created.

        Parameters
        ----------
        list_of_catalog
            filename of the catalogs from the configuration file
        workspace
            target folder where to save the catalogs if parquet format
        Returns
        -------
            list of the parquet files
        """
        catalog_files = []
        absolute_dir = os.getcwd()
        for file_id, original_file in enumerate(list_of_catalog["path"]):
            file_path = Path(original_file)
            name, suffix = file_path.stem, file_path.suffix
            target_file = Path(workspace) / (name + ".hdf5")
            if ".fits" == suffix:
                print("Converting fits catalogs to hdf5 format")
                vaex.open(
                    original_file, convert=str(target_file), progress=True
                )
            elif ".hdf5" == suffix:
                if not target_file.is_symlink():
                    original_file = os.path.abspath(os.path.join(absolute_dir, original_file))
                    os.symlink(original_file, target_file)
            else:
                raise ValueError(
                    f"Unrecognized file format. Expected fits or hdf5, got: {suffix}"
                )
            target_file = os.path.abspath(os.path.join(absolute_dir, target_file))
            catalog_files.append(str(target_file))
        return catalog_files

    def _get_column(self, col_name):
        """Returns column

        Build the columns as described in the config with a name or an evaluation function

        Parameters
        ----------
        col_name : str, dict
            Column name in the original catalog or dictionary with a
            description of how to build the column.

        Returns
        -------
        dask array
            DaskArray of the column
        """

        if isinstance(col_name, str):
            return self._df[col_name].to_dask_array()
        elif isinstance(col_name, dict):
            cat_tmp = {}
            expression = copy.copy(col_name["func"])
            for var_name in col_name["var"]:
                cat_tmp[var_name] = self._df[var_name].to_dask_array()
                expression = expression.replace(
                    f"${var_name}$", f"cat_tmp['{var_name}']"
                )
            try:
                res = eval(expression)
            except Exception as e:
                raise ValueError(
                    f"Error while evaluating expression: \n{expression}\n"
                    f"Got exception: \n{e}"
                )
            return res
