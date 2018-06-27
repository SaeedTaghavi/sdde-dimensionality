import luigi

def set_logging_level(level):
    luigi.interface.setup_interface_logging.has_run = False
    luigi.interface.setup_interface_logging(level_name=level)
