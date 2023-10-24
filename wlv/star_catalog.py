# Author: Axel Guinot (axel.guinot.astro@gmail.com)
# Star Catalogue

from .catalog import Catalog


class StarCatalog(Catalog):
    def __init__(self, config=None):
        super().__init__(config=config)

    def read_catalog(self):
        """Build the star catalog."""
        cat_config = self._config["star_catalog"]
        self._df = self._read_catalog(cat_config)
        # interprets columns
        self.column_names = list(cat_config["columns"].keys())
        self._columns = {}
        for column_name in self.column_names:
            self._columns[column_name] = self._get_column(
                cat_config["columns"][column_name],
            )
