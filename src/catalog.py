from astropy.io import fits

from src.parameters import Parameters


class Catalog:
    def __init__(self, **kwargs):

        self.pars = Parameters(required_pars=["filename"])
        # set default values

        # load parameters from user input:
        self.pars.update(kwargs)

        # TODO: verify all required inputs are present

    # def load_catalog(self, filename=None):
    #     if filename is None:
    #         # TODO: cite paper and explain how to download this file
    #         if self.project_name == "WD":
    #             filename = "GaiaEDR3_WD_main.fits"
    #         else:
    #             filename = self.project_name + "_catalog.fits"
    #
    #     path = "catalogs"
    #     self.catalog = Catalog(os.join(path, filename))
    #
    #     # with fits.open(os.path.join(path, filename)) as hdul:
    #     #     # read the table headers
    #     #     names = {}
    #     #     units = {}
    #     #     comments = {}
    #     #     # TODO: get this number from the header
    #     #     for i in range(161):
    #     #         names[i] = hdul[1].header.get(f'TTYPE{i + 1}')
    #     #         units[i] = hdul[1].header.get(f'TUNIT{i + 1}')
    #     #         comments[i] = hdul[1].header.get(f'TCOMM{i + 1}')
    #     #
    #     #     self.catalog = np.array(hdul[1].data)
