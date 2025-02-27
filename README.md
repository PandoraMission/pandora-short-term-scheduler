# Pandora Blank

This is a "[blank](https://en.wikipedia.org/wiki/Planchet)" repository which can be used to create tools within the Pandora ecosystem. This has all of the defaults that we require for Pandora software including requirements management with poetry and documentation.

If you are starting a new Pandora tool from scratch, consider forking this repository and updating it to begin your tool.

## Cloning the Repository

To get started, fork this repository on GitHub, then clone your fork:

```sh
  git clone https://github.com/your-username/pandora-blank.git
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

For more details, check the [Poetry documentation](https://python-poetry.org/docs/).

## Testing the package

The `Makefile` in this repository enables you to run tests, fix import order, lint the code and check for compliance with `isort`, `black`, `flake8`, `pytest`.

Before you push your updates, run `make` inside the directory and resolve any issues that arise. This repository includes actions that will run these tools on GitHub, and show you if they pass. (See the buttons below).

## Adding documentation

You should click the docs folder and read that README to see instructions on how to build docs.

## Updating the package version and publishing to pypi

The package version is automaticaly found from the `pyproject.toml` file. This means when you are ready to release a new version you only need to follow these steps:

1. Update the version number in the `pyproject.toml` file
2. Build the distribution using

    ```sh
    poetry build
    ```

3. Publish the package to pypi

    ```sh
    poetry publish
    ```

You should consider using the industry standard for updating version numbers which is major.minor.patch

- Major version changes indicate breaking changes that may not be backward-compatible.
- Minor version changes introduce new features in a backward-compatible manner.
- Patch version changes include bug fixes or small improvements that do not affect compatibility.

For example, if a package is at version 1.2.3, then:

- Increasing the major version to 2.0.0 means changes were introduced that break the API
- Increasing the minor version to 1.3.0 means new features were added while maintaining compatibility.
- Increasing the patch version to 1.2.4 means minor bug fixes or tweaks were made without changing functionality.

It is ok to release new versions often.

### Update repository settings

To use the actions and docs that are set up here you will need to update your repository settings. One of them is making sure docs are built from the `gh-pages` branch. You can update this in settings:

![gh-pages settings](docs/images/gh-pages.png)

To make sure that you can update the website automatically, you need to ensure the action has read and write permissions in Settings/Actions:

![gh-pages settings](docs/images/read-write.png)

I've set up dependabot in this repository to automatically update the github actions with the most recent versions. This will run every week. To make use of this you will need to add these repo settings

You should make a branch-ruleset for your main branch. Create a ruleset in the settings as shown below

![gh-pages settings](docs/images/branch-ruleset.png)

You will then need to go into the ruleset and

1. Name the ruleset (I called it main)
2. Make the ruleset active
3. Add people who can bypass the ruleset. I suggest you add the admins for the repo and organization

![gh-pages settings](docs/images/ruleset1.png)

Make sure you select these options:

1. Branch targeting criteria: Add the `default` branch
2. Turn on `Restrict deletions`
3. Turn on `Require a pull request before merging`
4. Turn off `Block force pushes`
5. Turn on `Require status checks to pass`

You should add any github actions you want to be required to have users pull requests be merged. I suggest you add flake8, black (`lint`), and the tests (`build 3.X`).

![gh-pages settings](docs/images/ruleset2.png)

## Setting up the README for your package

Your package must include a README document. You should use this file (`README.md`) as your README document. Your can delete all information above this line, and leave the information below as your README file. You should update the URLs in the badges below to be the URLs for your package on github. i.e. replace `https://github.com/pandoramission/pandora-blank/` with `https://github.com/username/reponame/`.

<a href="https://github.com/pandoramission/pandora-blank/actions/workflows/tests.yml"><img src="https://github.com/pandoramission/pandora-blank/workflows/tests/badge.svg" alt="Test status"/></a> <a href="https://github.com/pandoramission/pandora-blank/actions/workflows/black.yml"><img src="https://github.com/pandoramission/pandora-blank/workflows/black/badge.svg" alt="black status"/></a> <a href="https://github.com/pandoramission/pandora-blank/actions/workflows/flake8.yml"><img src="https://github.com/pandoramission/pandora-blank/workflows/flake8/badge.svg" alt="flake8 status"/></a> [![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://pandoramission.github.io/pandora-blank/)
[![PyPI - Version](https://img.shields.io/pypi/v/packagename)](https://pypi.org/project/packagename/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/packagename)](https://pypi.org/project/packagename/)

# Package Name

Include information about your package here.
