# SecML-Lib: A library for Secure Machine Learning

## Installation Guide
SecML-Lib should be installed in an existing or in a new
 [virtualenv](https://virtualenv.pypa.io) or [conda](https://conda.io)
 environment. System-wise installation is supported but discouraged and
 so will not be covered by this guide.

The setup process is managed by the python package `setuptools`. Be sure
 to have the latest version of the package installed in your env by
 calling `pip install -U setuptools`.

Once the environment is set up, SecML-Lib can installed and run by
 multiple means:
 1. Install from official PyPI repository **(not yet supported)**
    - `pip install secml-lib`
 2. Install from zip/wheel package:
    - `pip install <package-file>`
 3. Install from local GitLab repository:
    - Clone the project repository in a directory of your choice
    - Run installation as: `python setup.py install`
 4. Install from remote GitLab repository. In this case, given
    `{repourl}` in the format, es., `pragit.diee.unica.it/secml/secml-lib`:
    - `pip install git+ssh://git@{repourl}.git[@branch]#egg=secml-lib`
    A specific branch to install can be specified using `[@branch]` parameter.
    If omitted, the default branch will be installed.

In all cases, the setup process will try to install the correct dependencies.
In case something goes wrong during the install process, try to install
 the dependencies **first** by calling: `pip install -r requirements.txt`

SecML-Lib should now be importable in python via: `import secml`.

To update a current installation using any of the previous methods, add the 
 `-U` parameter after the `pip install` directive.

SecML-Lib can be added as a dependency for other libraries/project.
Just add `secml-lib` (**not yet supported**) or the full repository
path command `git+ssh://git@{repourl}.git[@branch]#egg=secml-lib` to
your `requirements.txt` file.

#### Editable Installation (development mode)

For SecML-Lib developers or users want to use the latest `dev` version
of the library, `pip` provides a convenient option which is called: **editable mode**.

By calling `pip install` with the `-e` option or `python setup.py develop`,
only a reference to the project files is "installed" in the active
environment. In this way, project files can be edited/updated and the
new versions will be automatically executed by the Python interpreted.

Two common scenario are listed below:
1. Editable install from already cloned local repository
    - Navigate to the repository directory
    - Run `python setup.py develop`
2. Editable install from remote repository
    - Run `pip install -e git+ssh://git@{repourl}.git[@branch]#egg=secml-lib`
    - Project will be cloned automatically in `<venv path>/src/secml-lib`
    - The new repository can then be updated using standard `git` commands

Editable installs are also available while using SecML-Lib as a
dependency of other libraries/projects (see "Installation Guide"
section for more informations).
