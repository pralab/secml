"""
.. module:: FileManager
   :synopsis: A collection of useful methods for directories and files managing.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import os
import shutil
import tempfile

# Remember to add any new method to following list
__all__ = ['folder_exist', 'file_exist', 'make_folder_incwd', 'make_folder',
           'remove_folder', 'make_rand_folder', 'abspath', 'normpath',
           'join', 'split', 'expanduser', 'dirsep', 'get_tempfile']


def folder_exist(folder_path):
    """Test whether a folder exists.

    Returns False for broken symbolic links.

    """
    return True if os.path.isdir(folder_path) else False


def file_exist(file_path):
    """Test whether a file exists.

    Returns False for broken symbolic links.

    """
    return True if os.path.isfile(file_path) else False


def make_folder_incwd(folder_name, mode=0o777):
    """Create a directory named folder_name inside current working directory (cwd).

    Parameters
    ----------
    folder_name : str
        Desired name for the new folder.
    mode : oct, optional
        Octal literal representing the numeric mode to use.
        On some systems, mode is ignored. Where it is used,
        the current umask value is first masked out.
        If bits other than the last 9 (i.e. the last 3 digits
        of the octal representation of the mode) are set,
        their meaning is platform-dependent. On some platforms,
        they are ignored and you should call chmod() explicitly
        to set them. Default `0o777`.

    See Also
    --------
    make_folder : Create a directory given full path.

    """
    return make_folder(os.path.join(os.path.dirname(os.getcwd()), folder_name), mode=mode)


def make_folder(folder_path, mode=0o777):
    """Create a directory inside folder_path with numeric mode 'mode'.

    All intermediate-level directories needed to contain
    the leaf directory will be recursively made.

    Parameters
    ----------
    folder_path : str
        Desired path for the new folder.
    mode : oct, optional
        Octal literal representing the numeric mode to use.
        On some systems, mode is ignored. Where it is used,
        the current umask value is first masked out.
        If bits other than the last 9 (i.e. the last 3 digits
        of the octal representation of the mode) are set,
        their meaning is platform-dependent. On some platforms,
        they are ignored and you should call chmod() explicitly
        to set them. Default `0o777`.

    See Also
    --------
    make_folder_inpath : Create a directory inside a specific folder.

    """
    os.makedirs(os.path.abspath(folder_path), mode)  # mkdir will manage errors
    return folder_path


def remove_folder(folder_path, force=False):
    """Remove (delete) the directory path.

    Path must point to a directory (but not a symbolic
    link to a directory).

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to folder to remove.
    force : bool, optional
        By default, if force is False, directory is removed only
        if empty, otherwise, OSError is raised. Set to True in
        order to remove the whole directory and its subdirectories.

    """
    if force is False:
        try:  # rmdir does not remove the directory if not empty
            os.rmdir(os.path.abspath(folder_path))
        except OSError as e:
            raise OSError("{:}. Try use force=True next time.".format(e))
    else:  # rmtree remove current directory entirely
        shutil.rmtree(os.path.abspath(folder_path))


def remove_file(file_path):
    """Remove (delete) target file.

    If path is a directory, OSError is raised (see `.remove_folder`).
    On Windows, attempting to remove a file that is in
    use causes an exception to be raised; on Unix,
    the directory entry is removed but the storage
    allocated to the file is not made available until
    the original file is no longer in use.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to file to remove.

    """
    os.remove(os.path.abspath(file_path))


# TODO: ADD DOCSTRING
def ignore_function(ignore):
    def _ignore_(path, names):
        ignored_names = []
        if ignore in names:
            ignored_names.append(ignore)
        return set(ignored_names)

    return _ignore_


def copy_folder(folder_path, copy_folder_path, ignore_file=''):
    """Copy a folder and every folder/file it contain. 

    Path must point to a directory (can not be a symbolic link).

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to folder to copy.
    copy_folder_path : str
        Absolute or relative path of new folder where you want store folder_path's data.
    ignore_file : tuple 
        contain folder name of file + path that we wouldn't copy 
        for example :
            file_to_ignore = ('filetoignore.txt','foldertoignore')
    """
    if not os.path.isdir(folder_path):
        raise OSError("folder that you want copy doesn't exist!")
    else:
        try:
            shutil.copytree(folder_path, copy_folder_path, ignore=shutil.ignore_patterns(*ignore_file))
        except OSError as e:
            print('Directory not copied. Error: %s' % e)


# TODO: CHECK DOCSTRING
def copy_file(file_path, copy_file_path):
    """Copy one file. 

    Path must point to a file (can not be a symbolic link).

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to folder to copy.
    copy_folder_path : str
        Absolute or relative path of new folder where you want store folder_path's data.
    """
    if not os.path.isfile(file_path):
        raise OSError("file that you want copy doesn't exist!")
    else:
        try:
            shutil.copy(file_path, copy_file_path)
        except OSError as e:
            print('File not copied. Error: %s' % e)


def make_rand_folder(folder_path=None, custom_name=None):
    """Create a random named folder.

    Random name will be selected inside the integer range
    between 1 and 1 million [1, 1kk).

    Parameters
    ----------
    folder_path : str, optional
        Path where to create the new directory. If None, folder
        will be created inside calling file folder.
    custom_name : str, optional
        Custom name to add before the random ID number. An underscore
        is placed between ID and custom_name.

    Returns
    -------
    target_path : str
        Absolute path of created directory.

    Notes
    -----
    There is a small chance that randomly generated folder
    already exists. Just run the function again :)

    """
    # Generating random folder ID
    from numpy import random
    folder_id = random.randint(1, 1000000)
    folder_name = str(folder_id) if custom_name is None else "{:}_{:}".format(custom_name, folder_id)
    # make_folder will manage errors
    return make_folder_incwd(folder_name) if folder_path is None else make_folder(join(folder_path, folder_name))


def abspath(file_name):
    """Return the absolute path to file.

    File name, as well as directory separator, is not
    added to the end of the returned path.

    Examples
    --------
    >>> import secml.utils.c_file_manager as fm

    >>> cur_file = fm.split(__file__)[1]  # Getting only the filename
    >>> cur_file  # doctest: +SKIP
    'c_folder_manager.py'
    >>> fm.abspath(cur_file)[-12:]
    '/secml/utils'

    """
    return os.path.dirname(os.path.abspath(file_name))


def normpath(path):
    """Normalize a pathname.

    Normalize a pathname by collapsing redundant separators and
    up-level references so that A//B, A/B/, A/./B and A/foo/../B
    all become A/B. This string manipulation may change the meaning
    of a path that contains symbolic links. On Windows, it converts
    forward slashes to backward slashes.

    Examples
    --------
    >>> import secml.utils.c_file_manager as fm

    >>> cur_path = fm.split(__file__)[0]  # Getting only the filename
    >>> cur_path  # doctest: +SKIP
    '---/src/secml/utils'
    >>> upper_path = fm.join(cur_path, '..', '..')
    >>> upper_path  # doctest: +SKIP
    '---/src/secml/utils/../..'
    >>> fm.normpath(upper_path)  # doctest: +SKIP
    '---/src'

    """
    return os.path.normpath(path)


def join(*paths):
    """Join one or more path components intelligently.

    The return value is the concatenation of path and any
    members of `*paths` with exactly one directory separator
    (os.sep) following each non-empty part except the last,
    meaning that the result will only end in a separator if
    the last part is empty. If a component is an absolute path,
    all previous components are thrown away and joining
    continues from the absolute path component.

    See Also
    --------
    split : Split the pathname path into a pair (head, tail).

    """
    return os.path.join(*paths)


def split(path):
    """Split the pathname path into a pair (head, tail).

    Tail is the last pathname component and head is everything leading
    up to that. The tail part will never contain a slash; if path ends
    in a slash, tail will be empty. If there is no slash in path, head
    will be empty. If path is empty, both head and tail are empty.
    Trailing slashes are stripped from head unless it is the root (one
    or more slashes only). In all cases, join(head, tail) returns a path
    to the same location as path (but the strings may differ).

    Returns
    -------
    out_split : tuple of str
        A tuple of strings consisting of (head, tail), where tail
        is the last pathname component and head is everything leading
        up to that.

    See Also
    --------
    join : Join one or more path components intelligently.

    Examples
    --------
    >>> import secml.utils.c_file_manager as fm

    >>> path = fm.join('dir1','dir2','dir3')
    >>> path
    'dir1/dir2/dir3'
    >>> print(fm.split(path))
    ('dir1/dir2', 'dir3')

    """
    return os.path.split(path)


def splitext(path):
    """Split the pathname path into a pair (root, ext).

    Content of `ext` is empty or begins with a period
    and contains at most one period.

    Leading periods on the basename are ignored;
    splitext('.zip') returns ('.zip', '').

    Return
    ------
    out_splitext : tuple of str
        A tuple of strings consisting of (root, ext),
        such that root + ext == path

    Examples
    --------
    >>> import secml.utils.c_file_manager as fm

    >>> path = fm.join('dir1','file.gz')
    >>> path
    'dir1/file.gz'
    >>> print(fm.splitext(path))
    ('dir1/file', '.gz')

    >>> path = fm.join('dir1','file.tar.gz')
    >>> path
    'dir1/file.tar.gz'
    >>> print(fm.splitext(path))  # Only the first (real) extension is returned
    ('dir1/file.tar', '.gz')

    """
    return os.path.splitext(path)


def expanduser(path):
    """Replace user path shortcut with real user path.

    On Unix and Windows, return `path` with an initial ~ or ~user replaced
    by that user's home directory.

    On Unix, an initial ~ is replaced by the environment variable HOME
    if it is set; otherwise the current user's home directory is looked
    up in the password directory through the built-in module pwd. An initial
    ~user is looked up directly in the password directory.

    On Windows, HOME and USERPROFILE will be used if set, otherwise a
    combination of HOMEPATH and HOMEDRIVE will be used. An initial ~user is
    handled by stripping the last directory component from the created user
    path derived above.

    If the expansion fails or if the path does not begin with a tilde, the
    path is returned unchanged.

    Examples
    --------
    >>> import secml.utils.c_file_manager as fm

    >>> fm.expanduser('~')  # doctest: +SKIP
    '/home/username'
    >>> fm.expanduser(fm.join('~','documents'))  # doctest: +SKIP
    '/home/username/documents'

    """
    return os.path.expanduser(path)


def listdir(path):
    """Return list of elements inside directory.

    Return a list containing the names of the entries in the
    directory given by path. The list is in arbitrary order.
    It does not include the special entries '.' and '..' even
    if they are present in the directory.

    """
    return os.listdir(path)


def dirsep():
    """The character used by the operating system to separate
    pathname components. This is '/' for POSIX and '\\' for Windows.
    Note that knowing this is not sufficient to be able to parse or
    concatenate pathnames, use CFileManager.split() and
    CFileManager.join() instead, but it is occasionally useful.

    """
    return os.sep


def get_tempfile():
    """Returns an handle to a temporary file.

    The file will be destroyed as soon as it is closed
    (including an implicit close when the object is
    garbage collected).

    """
    return tempfile.TemporaryFile()
