# SecML-Lib: A library for Secure Machine Learning

## Installation Guide
SecML-Lib should be installed as first choice in an existing or in a new
 [virtualenv](https://virtualenv.pypa.io) or [conda](https://conda.io)
 environment. System-wise installation is supported but discouraged and
 so will not be covered by this guide.

The setup process is managed by the python package `setuptools`. Be sure
 to have the latest version of the package installed in your env by
 calling `pip install -U setuptools`.

Once the environment is set up, SecML-Lib can installed and run by
 multiple means:
 1. Install from remote GitLab repository:
    - Clone the project repository in a directory of your choice
    - Run installation as: `python setup.py install`
 2. Install from remote GitLab repository. In this case, given
    `{repourl}` in the format, es., `pragit.diee.unica.it/secml/secml-lib`:
    - `pip install git+ssh://git@{repourl}.git[@branch]#egg=secml-lib`
    A specific branch to install can be specified using `[@branch]` parameter.
    If omitted, the default branch will be installed.
 3. Install from zip/wheel package:
    - `pip install {package}`
 4. Install from official PyPI repository **(not yet supported)**
    - `pip install secml-lib`

In all cases, the setup process will try to install the correct dependencies.
In case something goes wrong during the install process, try to install
 the dependencies **first** by calling: `pip install -r requirements.txt`

SecML-Lib should now be importable in python via: `import secml`.

To update a current installation using any of the previous methods, add the 
 `-U` parameter after the `pip install` directive.