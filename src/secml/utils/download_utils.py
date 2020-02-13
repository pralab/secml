"""
.. module:: DownloadUtils
   :synopsis: Collection of mixed utilities for downloading files

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import sys
import re
import requests
import hashlib
from urllib import parse

from secml.utils import fm

__all__ = ['dl_file', 'dl_file_gitlab', 'md5']


def dl_file(url, output_dir, user=None, headers=None,
            chunk_size=1024, md5_digest=None):
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
    headers : dict or None, optional
        Dictionary with any additional header for the download request.
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

    r = requests.get(url, auth=auth, headers=headers, stream=True)

    if r.status_code != 200:
        raise RuntimeError(
            "File is not available (error code {:})".format(r.status_code))

    # Get file size (bytes)
    if "content-length" in r.headers:
        total_size = r.headers.get('content-length').strip()
        total_size = int(total_size)
    else:  # Total size unknown
        total_size = None

    dl = 0

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1 byte")

    sys.stdout.write("Downloading from `{:}`".format(url))
    if total_size is not None:
        sys.stdout.write(" ({:} bytes)".format(total_size))
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Create output directory if not exists
    if not fm.folder_exist(output_dir):
        fm.make_folder(output_dir)

    try:  # Get the filename from the response headers
        fname = re.findall(
            r"filename=\"(.+)\"", r.headers["Content-Disposition"])[0]
    except (KeyError, IndexError):
        # Or use the last part of download url (removing parameters)
        fname = url.split('/')[-1].split('?', 1)[0]

    # Build full path of output file
    out_path = fm.join(output_dir, fname)

    # Read data and store each chunk
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # Report progress (if total_size is known)
                if total_size is not None:
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


def dl_file_gitlab(repo_url, file_path, output_dir, branch='master',
                   token=None, chunk_size=1024, md5_digest=None):
    """Download file from a gitlab.com repository and store in output_dir.

    Parameters
    ----------
    repo_url : str
        Url of the repository from which download the file.
        Can include the `http(s)://gitlab.com/` prefix.
    file_path : str
        Path to the file to download, relative to the repository.
    output_dir : str
        Path to the directory where the file should be stored.
        If folder does not exists, will be created.
    branch : str, optional
        Branch from which the file should be downloaded. Default 'master'.
    token : str or None, optional
        Personal access token, required to access private repositories.
        See: https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html
    chunk_size : int, optional
        Size of the data chunk to read from url in bytes. Default 1024.
    md5_digest : str or None, optional
        Expected MD5 digest of the downloaded file.
        If a different digest is computed, the downloaded file will be
        removed and ValueError is raised.

    """
    # Url of Repository files API, to be populated later
    api_url = 'https://gitlab.com/api/v4/projects/' \
              '{:}/repository/files/{:}/raw?ref={:}'

    # Decode the repository url by removing 'gitlab.com' prefix if defined
    # To make urlparse work correctly, we should add a '//gitlab.com/' prefix
    if repo_url.startswith('gitlab.com'):  # Handle 'gitlab.com/REPO' case
        repo_url = '//' + repo_url
    if not repo_url.startswith(
            ('https://gitlab.com', 'http://gitlab.com', '//gitlab.com')):
        # Handle the '/REPO/' case by stripping the first slash (if any)
        repo_url = '//gitlab.com/' + repo_url.lstrip('/')
    # Strip last slash (if any) and parse
    repo_url_parsed = parse.urlparse(repo_url.rstrip('/'))
    # Remove the first slash always left by urlparse and encode
    repo_url_encoded = parse.quote(repo_url_parsed.path[1:], safe='')

    # Strip the first slash (if any) and encode the file path
    file_path_encoded = parse.quote(file_path.lstrip('/'), safe='')

    # Build the final download url
    url = api_url.format(repo_url_encoded, file_path_encoded, branch)

    # Pass the private token as a request's header if defined
    headers = {'PRIVATE-TOKEN': token} if token is not None else None

    return dl_file(url, output_dir, headers=headers,
                   chunk_size=chunk_size, md5_digest=md5_digest)


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
