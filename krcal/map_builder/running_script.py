from invisible_cities.core.configure import configure
from krcal.map_builder.map_builder_functions import automatic_test
import sys

if __name__ == "__main__":
    config = configure(sys.argv).as_namespace
    automatic_test(config)
