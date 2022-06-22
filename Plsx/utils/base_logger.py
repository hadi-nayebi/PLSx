import logging
from os.path import dirname
from pathlib import Path

from Plsx.utils.file_manager import get_root


class Logger:

    root = get_root(__file__, retrace=2)
    levels = {"info": logging.INFO}

    def __init__(self, auto_config=True):
        self.logger = logging
        if auto_config:
            log_dir = self.root / "logs"
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            self.set_basic_config()

    def set_basic_config(self, log_file="base_logger.log", level="info"):
        self.logger.basicConfig(
            filename=self.root / "logs" / log_file,
            filemode="a",
            format="%(levelname)s:%(asctime)s - %(message)s",
            level=self.levels[level],
        )


# use this for base logger access
base_logger = Logger()
