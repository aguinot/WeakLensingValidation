from wlv.config_parser import ConfigParser
from wlv.star_catalog import StarCatalog
from wlv.galaxy_catalog import GalaxyCatalog
from wlv.psf_residuals import PSFResiduals

config_file = "./config_samples/lensfit.yaml"

c = ConfigParser(config_file)
print(f"Done interpreting {config_file}")
cat_galaxy = GalaxyCatalog(config=c)
print("Done loading galaxies catalog.")
cat_star = StarCatalog(config=c)
print("Done loading stars catalog.")

psf_res = PSFResiduals(star_cat=cat_star, config=c)
print("Ready to compute psf residuals")
psf_res.process()
print("All is done.")