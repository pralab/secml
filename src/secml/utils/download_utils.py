"""
.. module:: DownloadUtils
   :synopsis: Collection of mixed utilities for downloading files

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from __future__ import division
import sys
import requests
import hashlib
from io import open  # TODO: REMOVE AFTER TRANSITION TO PYTHON 3

from secml.utils import fm


def dl_file(url, output_dir, user=None, chunk_size=1024, md5_digest=None):
    """Download file from input url and store in output_dir.

    Parameters
    ----------
    url : str
        Url of the file to download.
    output_dir : str
        Path to the directory where the file should be stored.
        If folder does not exists, will be created.
    user : str or None, optional
        String with the user[:password] if required for accessing url.
    chunk_size : int, optional
        Size of the data chunk to read from url in bytes. Default 1024.
    md5_digest : str or None, optional
        Expected MD5 digest of the downloaded file.
        If a different digest is computed, the downloaded file will be
        removed and ValueError is raised.

    """
    # Parsing user string
    auth = tuple(user.split(':')) if user is not None else None
    # If no password is specified, use an empty string
    auth = (auth[0], '') if auth is not None and len(auth) == 1 else auth

    r = requests.get(url, auth=auth, stream=True)

    if r.status_code != 200:
        raise RuntimeError(
            "File is not available (error code {:})".format(r.status_code))

    # Get file size (bytes)
    total_size = r.headers.get('content-length').strip()
    total_size = int(total_size)
    dl = 0

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1 byte")

    sys.stdout.write(
        "Downloading from `{:}` ({:} bytes)\n".format(url, total_size))
    sys.stdout.flush()

    # Create output directory if not exists
    if not fm.folder_exist(output_dir):
        fm.make_folder(output_dir)

    # Build full path of output file
    out_path = fm.join(output_dir, url.split('/')[-1])

    # Read data and store each chunk
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # Report progress
                dl += len(chunk)
                done = int((50 * dl) / total_size)
                if sys.stdout.isatty() is True:
                    # Provide real-time updates (if stdout is a tty)
                    sys.stdout.write("\r[{:}{:}] {:}/{:}".format(
                        '=' * done, ' ' * (50-done), dl, total_size))
                    sys.stdout.flush()

    sys.stdout.write("\nFile stored in `{:}`\n".format(out_path))
    sys.stdout.flush()

    if md5_digest is not None and md5_digest != md5(out_path, chunk_size):
        fm.remove_file(out_path)  # Remove the probably-corrupted file
        raise ValueError("Unexpected MD5 hash for the downloaded file.")

    return out_path


def md5(fname, blocksize=65536):
    """Generate RSA's MD5 digest for input file.

    Parameters
    ----------
    fname : str
        Path to the file to parse
    blocksize : int
        Size in bytes of the file chunks to read. Default 65536.

    Returns
    -------
    str
        MD5 hex digest of input file.

    """
    hash_md5 = hashlib.md5()
    with open(fname, mode='rb') as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
