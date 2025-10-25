# Pandora Blank

This is a "[blank](https://en.wikipedia.org/wiki/Planchet)" repository which can be used to create tools within the Pandora ecosystem. This has all of the defaults that we require for Pandora software including requirements management with poetry and documentation.

If you are starting a new Pandora tool from scratch, consider forking this repository and updating it to begin your tool.

## Cloning the Repository

To get started, fork this repository on GitHub, then clone your fork:

```sh
  git clone https://github.com/pandoramission/pandora-blank.git
  cd pandora-blank
```

## Updating the Repository with a New Package Name

1. Rename the `packagename` directory inside `src/` to your new package name.
2. Update `pyproject.toml`:
   - Change `name = "packagename"` to `name = "your_new_package"`.
3. Update `mkdocs.yml`:
   - Change `site_name: packagename` to `site_name: your_new_package`.
4. Update all imports in the codebase from `packagename` to `your_new_package`.

## Using the Config File System

The package includes a configuration system that loads and saves user preferences.

The configuration file is stored in a directory that users should be able to access invariant of what sort of machine they are using.

### Setting default configuration

You can update the default configuration for yorur package by updating the `reset_config` function with your defaults. For example, here is the function from one of our existing tools `pandorapsf`:

   ```python
   def reset_config():
       config = configparser.ConfigParser()
       config["SETTINGS"] = {
           "log_level": "INFO",
           "data_dir": user_data_dir("pandorapsf"),
           "vis_psf_download_location": "https://zenodo.org/records/11228523/files/pandora_vis_2024-05.fits?download=1",
           "nir_psf_download_location": "https://zenodo.org/records/11153153/files/pandora_nir_2024-05.fits?download=1",
           "vis_psf_creation_date": "2024-05-14T11:38:14.755119",
           "nir_psf_creation_date": "2024-05-08T15:02:58.461202",
       }
   
       with open(CONFIGPATH, "w") as configfile:
           config.write(configfile)
   ```

This sets the default settings, but users can always update the settings themselves if they choose to.
Settings can be reset to defaults by the user at any time using:

  ```python
  from packagename import reset_config
  reset_config()
  ```

If you update default settings, users will have to reset their config to get these changes.

This repository sets up a default log level and a default `data_dir`. This is a directory that you should use to store any package data that needs to be downloaded to run the package. For example, if you needed large PSF files, or a large database, you would put that in this `data_dir`. This folder will be shared by any install of your package on a given machine. This means that multiple installs of a given package will not require redownloading any files in this directory.

### Updating configuration as a user

Your users can load the configuration with:

  ```python
  from packagename import load_config
  config = load_config()
  ```

If they change the configuration they can save it using

  ```python
  from packagename import save_config
  config["SETTINGS"]["log_level"] = "INFO"
  save_config(config)
  ```

Users can find where the configuration file is stored using

  ```python
  from packagename import CONFIGDIR
  print(CONFIGDIR)
  ```

## Using the Logging System

The package includes a logging system using `RichHandler` for formatted console output. If you clone this repo, you will automatically have a logger. You can use it like this:

  ```python
  from packagename import logger
  logger.info("This is an info message")
  logger.warning("This is a warning message")
  ```

The default logger level is set in the config file, but you can update this in your session using the following, where I set the logger level to `"INFO"`.

  ```python
  from packagename import logger
  logger.setLevel("INFO")
  ```

## Managing Dependencies with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

Make sure you have poetry installed on your machine

  ```sh
  pip install --upgrade poetry
  ```

After you clone the repository and update to your new package name you will need to install the package using

  ```sh
  poetry install
  ```

(If you do not update the package name you will install a python package called `packagename`.)

To add new dependencies to your package you can use

  ```sh
  poetry add dependency
  ```

- Add a development dependency:

  ```sh
  poetry add --group dev dependency
  ```

Development dependencies are only needed for development and are not included when installing the package in production. Add development dependencies for things like testing and building docs.

If you are using the package and need these development dependencies you should use

  ```sh
  poetry install --with dev
  ```

If you think your versions of packages are out of date you can run

  ```sh
  poetry update
  ```

Poetry installs unique copies of your dependencies so you can work on multiple packages without sharing versions of dependencies. Therefore, to work within this ecosystem you must use `poetry run` to run anything, e.g.

Instead of using

  ```sh
  python
  ```

You would use

  ```sh
  poetry run python
  ```

Instead of using

  ```sh
  jupyter-lab
  ```

Use

  ```sh
  poetry run jupyter-lab
  ```
# Pandora Short-Term Scheduler (shortschedule)

shortschedule is a compact toolkit for parsing, processing, and visualizing Pandora
science calendars (PAN-SCICAL XML). It was created to help adjust observation
timing and payload parameters when updated spacecraft TLEs are available.

Key features
------------

- Parse PAN-SCICAL XML into Python data models (`ScienceCalendar`, `Visit`, `ObservationSequence`).
- Process calendars with updated TLEs to fill visibility gaps, extend/shrink sequences,
  and recompute payload integration parameters for VIS/NIR payloads.
- Export processed calendars back to PAN-compliant XML using `XMLWriter`.
- Produce Gantt charts and summary visualizations for schedule comparison.

Installation
------------

Recommended: use Poetry to create an isolated environment and install dev dependencies:

```bash
poetry install --with dev
```

If you prefer pip, install the runtime dependencies listed in `pyproject.toml`.

Running tests and linters
-------------------------

Use the Makefile targets (these wrap Poetry in CI):

```bash
make pytest   # run tests
make flake8   # lint
make black    # format (Black line length = 79)
make isort    # sort imports
```

Or run pytest directly if your environment is already prepared:

```bash
pytest -q
```

Quick examples
--------------

Parse a calendar:

```python
from shortschedule import parse_science_calendar

cal = parse_science_calendar('data/PAN-SCICAL-example.xml')
```

Process a calendar using updated TLE lines:

```python
from shortschedule import ScheduleProcessor

proc = ScheduleProcessor(tle_line1, tle_line2)
processed = proc.process_calendar(cal, window_start='2025-10-01T00:00:00Z')
```

Write processed calendar to XML:

```python
from shortschedule import XMLWriter
XMLWriter().write_calendar(processed, output_path='processed.xml')
```

Developer notes
---------------

- Time types use `astropy.time.Time` and `TimeDelta` throughout the codebase — keep times
  as astropy objects when possible.
- Visibility checks rely on `pandoravisibility.Visibility`. Tests that exercise visibility
  should mock this dependency to keep tests deterministic.
- The models provide XML helpers to read and update payload parameters — prefer those helpers
  rather than manipulating raw XML strings.

Contributing
------------

Please run the tests and linters before opening pull requests. Add tests for any new
behavior and keep changes small and well-documented.

License
-------

See the repository LICENSE file for terms.
