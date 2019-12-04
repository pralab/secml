************
Contributing
************

**Your contribution is foundamental!**

If you want to help the development of SecML, just set up the project locally
by the following means:

 1. Install from local GitLab repository:

    - Clone the project repository in a directory of your choice
    - Run installation as: ``pip install .``

 2. Install from remote GitLab repository. In this case, given ``{repourl}``
    in the format, es., ``gitlab.com/secml/secml``:

    - ``pip install git+ssh://git@{repourl}.git[@branch]#egg=secml``
      A specific branch to install can be specified using ``[@branch]`` parameter.
      If omitted, the default branch will be installed.

Contributions can be sent in the form of a merge request via our
`GitLab issue tracker <https://gitlab.com/secml/secml/issues>`_.

SecML can also be added as a dependency for other libraries/project.
Just add ``secml`` or the full repository path command
``git+ssh://git@{repourl}.git[@branch]#egg=secml`` to the ``requirements.txt`` file.

Editable Installation (development mode)
----------------------------------------

For SecML developers or users want to use the latest ``dev`` version of
the library (soon available to the public), ``pip`` provides a convenient
option which is called: **editable mode**.

By calling ``pip install`` with the ``-e`` option or ``python setup.py develop``,
only a reference to the project files is "installed" in the active
environment. In this way, project files can be edited/updated and the
new versions will be automatically executed by the Python interpreter.

Two common scenarios are listed below:

 1. Editable install from a previously cloned local repository

    - Navigate to the repository directory
    - Run ``python setup.py develop``

 2. Editable install from remote repository

    - Run ``pip install -e git+ssh://git@{repourl}.git[@branch]#egg=secml``
    - Project will be cloned automatically in ``<venv path>/src/secml``
    - The new repository can then be updated using standard ``git`` commands

Editable installs are also available while using SecML as a
dependency of other libraries/projects
(see `Installation Guide <https://secml.gitlab.io/#installation-guide>`_ for more information).