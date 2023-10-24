# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Galaxy Catalogue

from .catalog import Catalog


class GalaxyCatalog(Catalog):
    def __init__(self, config=None):
        super().__init__(config=config)

    def read_catalog(self):
        """Build the galaxy catalog."""
        cat_config = self._config["galaxy_catalog"]
        self._df = self._read_catalog(cat_config)
        self.column_names = list(cat_config["columns"].keys())
        self._columns = {}
        for column_name in self.column_names:
            self._columns[column_name] = self._get_column(
                cat_config["columns"][column_name],
            )
