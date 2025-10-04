# Standard library
import logging  # noqa: E402
import os  # noqa
import time  # noqa: E402
from glob import glob
from threading import Event, Thread  # noqa: E402

# Third-party
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"
PANDORASTYLE = glob(f"{PACKAGEDIR}/data/pandora.mplstyle")

# Standard library
import configparser  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa

# Third-party
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from appdirs import user_config_dir, user_data_dir  # noqa: E402


def get_version():
    try:
        return version("packagename")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


# Custom Logger with Rich
class PandoraLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        console = Console()
        self.handler = RichHandler(
            show_time=False, show_level=False, show_path=False, console=console
        )
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.addHandler(self.handler)
        self.spinner_thread = None
        self.spinner_event = None

    def start_spinner(self, message="Processing..."):
        if self.spinner_thread is None:
            self.spinner_event = Event()
            self.spinner_thread = Thread(target=self._spinner, args=(message,))
            self.spinner_thread.start()

    def stop_spinner(self):
        if self.spinner_thread is not None:
            self.spinner_event.set()
            self.spinner_thread.join()
            self.spinner_thread = None
            self.spinner_event = None

    def _spinner(self, message):
        with self.handler.console.status(
            "[bold green]" + message
        ) as status:  # noqa
            while not self.spinner_event.is_set():
                time.sleep(0.1)


def get_logger(name="packagename"):
    """Configure and return a logger with RichHandler."""
    return PandoraLogger(name)


CONFIGDIR = user_config_dir("packagename")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")

logger = get_logger("packagename")


def reset_config():
    """Set the config to defaults."""
    # use this function to set your default configuration parameters.
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "log_level": "WARNING",
        "data_dir": user_data_dir("shortschedule"),
    }
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


def load_config() -> configparser.ConfigParser:
    """
    Loads the configuration file, creating it with defaults if it doesn't exist.

    Returns
    -------
    configparser.ConfigParser
        The loaded configuration.
    """

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIGPATH):
        # Create default configuration
        reset_config()
    config.read(CONFIGPATH)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves the configuration to the file.

    Parameters
    ----------
    config : configparser.ConfigParser
        The configuration to save.
    app_name : str
        Name of the application.
    """
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()

# Use this to check that keys you expect are in the config file.
# If you update the config file and think users may be out of date
# add the config parameters to this loop to check and reset the config.
for key in ["data_dir", "log_level"]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `packagename` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()

DATADIR = config["SETTINGS"]["data_dir"]
logger.setLevel(config["SETTINGS"]["log_level"])


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray(
                [(key, value) for key, value in dict(config[section]).items()]
            )
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)
