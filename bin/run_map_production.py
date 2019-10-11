from invisible_cities.core.configure import configure
from krcal.map_builder.map_builder_functions import map_builder
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.DEBUG)
this_script_logger = logging.getLogger(__name__)
this_script_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    config = configure(sys.argv).as_namespace
    map_builder(config)
