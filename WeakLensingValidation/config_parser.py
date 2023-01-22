# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# config parser
# Some of the ideas are inspired by the Galsim parser

import yaml
import copy
import re
import os

_main_fields = ['galaxy_catalog', 'star_catalog', 'mask_image']
_catalog_fields = ['type', 'path', 'columns']
_classic_gal_columns = [
    'ra',
    'dec',
    'e1',
    'e2',
    'weights',
    'e1_psf',
    'e2_psf',
    'mag'
]
_classic_star_columns = [
    'ra',
    'dec',
    'x',
    'y',
    'n_ccd',
    'e1_psf',
    'e2_psf',
    'size_psf',
    'e1_star',
    'e2_star',
    'size_star',
]


class ConfigParser():
    """ConfigParser

    Parse the input config file in .yaml format and build a dictionary.

    Parameters
    ----------
    config_path : str
        Path to the config file
    """

    def __init__(
        self,
        config_path=None,
    ):

        if isinstance(config_path, type(None)):
            raise ValueError("No config_path have been provided")
        if not isinstance(config_path, str):
            raise ValueError("config_path must be a string")
        if not os.path.exists(config_path):
            raise ValueError(f"No file found at: {config_path}")

        config_raw = self._read_yaml_file(config_path)

        self.parse_config(config_raw)

    def __str__(self):

        return yaml.dump(self.config)

    def __repr__(self):

        return self.__str__()

    def parse_config(self, config_raw):
        """parse config

        Parse the yaml dictionnay and transform the output in something the
        library can understand.

        Parameters
        ----------
            config_raw : dict
                raw output of the yaml loader
        """

        self.config = {}

        # Set workspace directory
        self._parse_workspace(config_raw)

        # First get the variables
        self._parse_variables(config_raw)

        # Galaxy catalogue
        if 'galaxy_catalog' in config_raw.keys():
            self._parse_galaxy_catalog(config_raw)
        # Star catalogue
        if 'star_catalog' in config_raw.keys():
            self._parse_star_catalog(config_raw)

    def _parse_workspace(self, config_raw):
        """parse workspace

        Setup the workspace directory and the name of the run. It also checks
        if the only_plot option is set.

        Parameters
        ----------
        config_raw : dict
            raw output of the yaml loader
        """

        # Check run name
        if 'run_name' not in config_raw.keys():
            raise ValueError("No run_name are provided")
        if not isinstance(config_raw['run_name'], str):
            raise ValueError("run_name must be a string")
        run_name = config_raw['run_name']

        # Check workspace
        if 'workspace_directory' not in config_raw.keys():
            raise ValueError("No workspace_directory are provided")
        if not isinstance(config_raw['workspace_directory'], str):
            raise ValueError("workspace_directory must be a string")
        workspace_dir_tmp = config_raw['workspace_directory']
        if not os.path.isdir(workspace_dir_tmp):
            raise ValueError(
                "The workspace path do not exist or is not a directory. "
                f"Got: {workspace_dir_tmp}"
            )
        workspace_dir = os.path.join(workspace_dir_tmp, run_name)

        if 'only_plot' in config_raw.keys():
            if not isinstance(config_raw['only_plot'], bool):
                raise ValueError(
                    "only_plot must in [True, Fasle]. "
                    f"Got: {config_raw['only_plot']}"
                )
            only_plot = config_raw['only_plot']
        else:
            only_plot = False

        if os.path.exists(workspace_dir):
            if only_plot:
                raise ValueError(
                    "When using only_plot, the workspace has to an already"
                    f"existing directory. Got: {workspace_dir}"
                )
        else:
            try:
                os.mkdir(workspace_dir)
            except Exception as e:
                raise ValueError(
                    "Error while creating the directory got the following "
                    f"exception:\n{e}"
                )

        config = {
            'path': workspace_dir,
            'run_name': run_name,
            'only_plot': only_plot,
        }

        config = {'workspace': config}
        self.config.update(config)

    def _parse_galaxy_catalog(self, config_raw):
        """parse the galaxy catalog

        Read and store informations about the galaxy catalog.

        Parameters
        ----------
            config_raw : dict
                raw output of the yaml loader
        """

        config = {}
        gal_dict = config_raw['galaxy_catalog']

        # Make sure the necessary information are provided
        if not all(
            [
                needed_key in gal_dict.keys()
                for needed_key in _catalog_fields
            ]
        ):
            raise ValueError(
                "The galaxy_catalog neeeds to have at least those entries: "
                f"{_catalog_fields}"
                )

        # Parse the path
        config['path'] = self._parse_path(gal_dict)

        if gal_dict["type"] == 'classic':
            config["type"] = 'classic'
            # Make sure the necessary columns are provided
            for needed_key in _classic_gal_columns:
                if needed_key not in gal_dict['columns'].keys():
                    raise ValueError(f"Column {needed_key} not provided")

            # Now we go through all columns
            # We cannot do it in the previous loop because more columns could
            # be given
            config["columns"] = {}
            for key in gal_dict['columns'].keys():
                column_tmp = gal_dict['columns'][key]

                # Assign internal naming to catalog naming
                if isinstance(column_tmp, str):
                    config["columns"][key] = column_tmp

                # Create a new column based on eval
                # Note that the evaluation is not done at this stage because
                # we don't have acces to the catalog yet. It will be complited
                # later when the catalog is actually read.
                elif isinstance(column_tmp, dict):
                    func, var = self._parse_eval(column_tmp, key)
                    config["columns"][key] = {
                        'func': func,
                        'var': var
                    }
        config = {'galaxy_catalog': config}
        self.config.update(config)

    def _parse_star_catalog(self, config_raw):
        """parse the star catalog

        Read and store informations about the star catalog.

        Parameters
        ----------
            config_raw : dict
                raw output of the yaml loader
        """

        config = {}
        star_dict = config_raw['star_catalog']

        # Make sure the necessary information are provided
        if not all(
            [
                needed_key in star_dict.keys()
                for needed_key in _catalog_fields
            ]
        ):
            raise ValueError(
                "The star_catalog neeeds to have at least those entries: "
                f"{_catalog_fields}"
                )

        # Parse the path
        config['path'] = self._parse_path(star_dict)
        # Checks if the star information are consitent with the galaxy catalog
        # if we have one
        if 'galaxy_catalog' in self.config.keys():
            if not isinstance(
                config['path']['path'],
                type(self.config['galaxy_catalog']['path']['path'])
            ):
                raise ValueError(
                    "The format of the star catalog is not consistant with "
                    "the galaxy catalog"
                )
            if isinstance(config['path']['path'], list):
                if (
                    len(config['path']['path']) !=
                    len(self.config['galaxy_catalog']['path']['path'])
                ):
                    raise ValueError(
                        "The number of star catalogs is not consistant with "
                        "the number of galaxy catalog"
                    )
            config['path']['keep_cat_history'] = \
                self.config['galaxy_catalog']['path']['keep_cat_history']
            config['path']['var_cat_history'] = \
                self.config['galaxy_catalog']['path']['var_cat_history']

        if star_dict["type"] == 'classic':
            config["type"] = 'classic'
            # Make sure the necessary columns are provided
            for needed_key in _classic_star_columns:
                if needed_key not in star_dict['columns'].keys():
                    raise ValueError(f"Column {needed_key} not provided")

            # Now we go through all columns
            # We cannot do it in the previous loop because more columns could
            # be given
            config["columns"] = {}
            for key in star_dict['columns'].keys():
                column_tmp = star_dict['columns'][key]

                # Assign internal naming to catalog naming
                if isinstance(column_tmp, str):
                    config["columns"][key] = column_tmp

                # Create a new column based on eval
                # Note that the evaluation is not done at this stage because
                # we don't have acces to the catalog yet. It will be complited
                # later when the catalog is actually read.
                elif isinstance(column_tmp, dict):
                    func, var = self._parse_eval(column_tmp, key)
                    config["columns"][key] = {
                        'func': func,
                        'var': var
                    }
        config = {'star_catalog': config}
        self.config.update(config)

    def _read_yaml_file(self, path):
        """read yaml file

        This method reads the input yaml file and return a raw dictionnary
        which will be parse after.

        Parameters
        ----------
            path : str
                path to the config file
        """

        self._config_path = path

        with open(path) as f:
            raw_dict = [c for c in yaml.load_all(f.read(), yaml.SafeLoader)]

        if len(raw_dict) != 1:
            raise ValueError(
                f"Error occured while reading config file at {path}"
            )

        return raw_dict[0]

    def _parse_variables(self, config_raw):
        """parse variable

        Parse the variables defined in config file.

        Parameters
        ----------
        config_raw : dict
            raw output of the yaml loader
        """

        # check if variables are defined
        var_keys = [key for key in config_raw.keys() if 'var' in key]

        if len(var_keys) != 0:
            self._var = {}
            for var_key in var_keys:
                if not isinstance(config_raw[var_key], dict):
                    raise ValueError(
                        f"Unroconized format for variable {var_key}"
                    )

                # Set the name of the variable
                if 'name' not in config_raw[var_key].keys():
                    raise ValueError(
                        f"No name found for varible {var_key}"
                    )
                if not isinstance(config_raw[var_key]['name'], str):
                    raise ValueError(
                        "Varibale name not of type string  for variable "
                        f"{var_key}"
                    )
                var_name = config_raw[var_key]['name']

                # Set the value fo the variable
                if 'value' not in config_raw[var_key].keys():
                    raise ValueError(
                        f"No value found for variable {var_key}"
                    )
                try:
                    var_val = eval(config_raw[var_key]['value'])
                except Exception as e:
                    raise ValueError(
                        f"Error while evaluating value of variable {var_key}:"
                        f"\n{e}"
                    )

                self._var[var_name] = var_val

    def _parse_path(self, cat_dict):
        """parse path

        Parse the path of a catalog.
        Path can be a list of paths or a function to evaluate. It is possible
        to keep track of which objects belong to which catalog to use that in
        the processing or the ploting.

        Parameters
        ----------
        cat_dict : dict
            Dictionnary containing information about a catalog

        Returns
        -------
        path_output: dict
            Output dictionnay with the information about the path of the
            catalog(s) folowing the format:
                - path: str or list
                    Path (or list of paths) of the catalog(s)
                - keep_cat_history: bool
                    Weither to keep track of objects. Only for multiple
                    catalogs
                - var_cat_history: list
                    list of int to keep the history of the catalogs. If not
                    set, a list is defined: [1, n_catalog]
        """

        path_output = {
            'path': '',
            'keep_cat_history': False,
            'var_cat_history': [],
        }

        # Get the path
        if isinstance(cat_dict['path'], str):
            path_output['path'] = cat_dict['path']
        elif isinstance(cat_dict['path'], list):
            path_output['path'] = cat_dict['path']
        elif isinstance(cat_dict['path'], dict):
            path_output['path'] = self._parse_eval(cat_dict['path'], 'path')
        else:
            raise ValueError("path must either a string, list or dict")

        # handle keep_cat_history
        if not (
            isinstance(cat_dict['path'], list)
            or isinstance(cat_dict['path'], dict)
        ):
            return path_output

        if 'keep_cat_history' in cat_dict.keys():
            path_output['keep_cat_history'] = \
                cat_dict['keep_cat_history']

        if not path_output['keep_cat_history']:
            return path_output

        if 'var_cat_history' in cat_dict.keys():
            var_name = cat_dict['var_cat_history']
            if not isinstance(var_name, str):
                raise ValueError(
                    f"var_cat_history not a string. Got: {var_name}"
                )
            if var_name not in self._var.keys():
                raise ValueError(
                    "Variable for var_cat_history not defined. Got: "
                    f"{var_name}"
                )
            var_history = self._var[var_name]
            if not isinstance(var_history, list):
                raise ValueError(
                    f"var_cat_history do not link to a list. Got: {var_name}"
                )
            if len(var_history) != len(path_output['path']):
                raise ValueError(
                    "Lenght of var_cat_history does not match number of "
                    "catalogs."
                    f"\nGot {len(var_history)} != {len(path_output['path'])}"
                )
            path_output['var_cat_history'] = var_history
            return path_output
        # We set a default list if not provided
        else:
            n_cat = len(path_output['path'])
            path_output['var_cat_history'] = list(range(1, n_cat+1))
            return path_output

    def _parse_eval(self, column, name):
        """parse eval

        Parse column or path that will use the eval function.
        Note that if the function contains variables, we first look for
        general variables defined in the config file and then among the
        catalog columns.

        Parameters
        ----------
            column : dict
                Dictionnary containing information about the eval
            name: str
                Name of variable to evaluate
        Returns
        -------
            res : float
                Result of eval fucntion
            func : str
                Function to evaluate with the column name to use from the
                catalog
            var_names: list
                List of the column names to replace in the function
        """
        if 'type' not in column.keys():
            raise ValueError(
                f"Unreconized type for {name}, missing type"
            )
        if column['type'].lower() != 'eval':
            raise ValueError(
                f"Unreconized type for {name}, "
                f"got {column['type']}"
            )
        if 'func' not in column.keys():
            raise ValueError(
                f"No function to evaluate for {name}"
            )

        # Copy the original string
        func = copy.copy(column["func"])

        # First check for variables
        if '$' in func:
            var_names = list(set(re.findall(r'\$(.*?)\$', func)))
            for var_name in var_names:
                if var_name in self._var.keys():
                    func = func.replace(
                        f"${var_name}$", f"{self._var[var_name]}"
                    )
                    var_names.remove(var_name)
            if len(var_names) == 0:
                try:
                    res = eval(func)
                except Exception as e:
                    raise ValueError(
                        f"Error while evaluating function: \n{func}\n"
                        f"Got exception: \n{e}"
                    )
                return res
            else:
                return func, var_names
        # Evaluate directly if no variables are found
        else:
            try:
                res = eval(func)
            except Exception as e:
                raise ValueError(
                    f"Error while evaluating function: \n{func}\n"
                    f"Got exception: \n{e}"
                )
            return res
