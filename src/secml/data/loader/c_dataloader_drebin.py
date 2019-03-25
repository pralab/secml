"""
.. module:: DataLoaderDrebin
   :synopsis: Loader the Drebin Android applications dataset

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import tarfile
from collections import OrderedDict
from six.moves import zip
from multiprocessing import Lock
import csv
import numpy as np

from secml.data.loader import CDataLoader, CDataLoaderSvmLight
from secml.data import CDataset
from secml.array import CArray
from secml.utils import fm
from secml.utils.download_utils import dl_file
from secml.utils.dict_utils import invert_dict
from secml import settings

DREBIN_URL = 'https://nue.diee.unica.it/public.php/webdav'
DREBIN_USER = 'e5a854edbdda1739f1f2bfbc31085e2b'
DREBIN_MD5 = 'ab73d749f2a8f51b30601ca7ed7c321b'

DREBIN_PATH = fm.join(settings.SECML_DS_DIR, 'drebin')

OBFUSCATION_MAPPING = {
    'ORIGINAL': 'ORIGINAL',
    'CLASS_ENCRYPTION': 'CLASS ENCRYPTION',
    'STRING_ENCRYPTION': 'STRING ENCRYPTION',
    'REFLECTION': 'REFLECTION',
    'TRIVIAL': 'TRIVIAL',
    'TRIVIAL_STRING_ENCRYPTION': 'TRIVIAL STRING E',
    'TRIVIAL_STRING_ENCRYPTION_REFLECTION': 'TRIVIAL STRING E + R',
    'TRIVIAL_STRING_ENCRYPTION_REFLECTION_CLASS_ENCRYPTION':
        'TRIVIAL STRING E + R + CLASS E'
}

FEAT_FAMILY_MAPPING = OrderedDict([
    ("Hardware components", ["features"]),
    ("Requested permissions", ["req_permissions"]),
    ("App components", ["activities", "services", "receivers", "providers"]),
    ("Filtered Intents", ["intent_filters"]),
    ("API calls", ["native_calls", "api_calls", "crypto_calls", "net_calls",
                   "telephony_calls", "reflection_calls","dynamic_calls"]),
    ("Used permissions", ["used_permissions"]),
    ("Suspicious API calls", ["suspicious_calls"]),
    ("Network addresses", ["urls"])
])
FEAT_FAMILY_MAPPING_INVERTED = invert_dict(FEAT_FAMILY_MAPPING)

DEXCODE_FAMILIES = ["API calls", "Used permissions",
                    "Suspicious API calls", "Network addresses"]
MANIFEST_FAMILIES = ["Hardware components", "Requested permissions",
                     "App components", "Filtered Intents"]


class CDataLoaderDrebin(CDataLoader):
    """Loads the Drebin Android applications dataset.

    Available at: https://www.sec.cs.tu-bs.de/~danarp/drebin/download.html

    Attributes
    ----------
    class_type : 'drebin'

    """
    __class_type = 'drebin'
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    # Dataset specific infos
    FEAT_FAMILY_MAPPING = FEAT_FAMILY_MAPPING
    FEAT_FAMILY_MAPPING_INVERTED = FEAT_FAMILY_MAPPING_INVERTED
    DEXCODE_FAMILIES = DEXCODE_FAMILIES
    MANIFEST_FAMILIES = MANIFEST_FAMILIES
    OBFUSCATION_MAPPING = OBFUSCATION_MAPPING

    def __init__(self):

        # Path dataset files
        self.data_path = fm.join(DREBIN_PATH, 'complete', 'all.libsvm')
        self.feat_freqs_path = fm.join(DREBIN_PATH, 'feat_freqs.txt')
        self.feat_mapping_path = fm.join(DREBIN_PATH, 'feature_mapping.txt')
        self.families_path = fm.join(DREBIN_PATH, 'sha256_family.csv')

        with CDataLoaderDrebin.__lock:
            # Download (if needed) data and extract it
            if not fm.file_exist(self.data_path):
                self._get_data(DREBIN_URL, DREBIN_PATH)

    def load(self, feats_info=False):
        """Load the dataset.

        The dataset consists of 126944 samples, 121329 benign applications
        and 5615 malware applications. Each sample is in sparse format,
        consisting of 1227080 binary (0-1) features each.

        Custom CDataset attributes:
         - 'infos': CArray with the 'hash.label' string for each point
        (only if `feats_info` is True)
         - 'original_idx': original index of each feature
         - 'feat_family_idx': for each feature, the index of the family
                relative to the CDataLoaderDrebin.FEAT_FAMILY_MAPPING
         - 'feat_desc': dict with the description of each original feature
         - 'app_family_map': dict with the id of each app family (0 is Benign,
                others are malware)
         - 'mal_family': dict with the app family id for
                each malware (hash). The hash can be retrieved from 'infos'.

        Parameters
        ----------
        feats_info : bool, optional
            If True, also load informations about the features.
            See custom attributes for a list of available data.
            Default False.

        Returns
        -------
        CDataset
            The Drebin dataset in sparse format.

        """
        self.logger.info(
            "Loading Drebin dataset from {:}".format(self.data_path))

        ds = CDataLoaderSvmLight().load(self.data_path,
                                        dtype_samples=int,
                                        dtype_labels=int,
                                        load_infos=True)

        if feats_info is False:  # Features data will not be loaded
            return ds

        self.logger.info(
            "Loading feats freqs from {:}".format(self.feat_freqs_path))

        feat_freqs = np.genfromtxt(
            self.feat_freqs_path, delimiter=',', dtype="|S500", autostrip=True)
        # Original index of each feature
        original_idx = CArray(feat_freqs[:, 1].astype(int))

        # Let's assign to each feature the index of the corresponding family
        # relative to the FEAT_FAMILY_MAPPING ordered dict
        feat_family = CArray(feat_freqs[:, 2])
        feat_family_idx = CArray.empty(feat_family.size, dtype=int)
        for f_idx, feat_fam in enumerate(feat_family):
            fam_inv = FEAT_FAMILY_MAPPING_INVERTED[feat_fam]
            feat_family_idx[f_idx] = list(FEAT_FAMILY_MAPPING).index(fam_inv)

        self.logger.info(
            "Loading feats mappings from {:}".format(self.feat_mapping_path))

        feat_mapping = np.genfromtxt(
            self.feat_mapping_path, delimiter=',', dtype="|S500",
            autostrip=True, skip_header=1)

        # Now create a dictionary with features ORIGINAL index as key
        # and the corresponding description as value
        feat_desc_idx = CArray(feat_mapping[:, 0].astype(int))
        feat_desc_str = CArray(feat_mapping[:, 1])
        feat_desc = {k: v for k, v in zip(feat_desc_idx, feat_desc_str)}

        self.logger.info(
            "Loading malware families from {:}".format(self.families_path))

        # Create dictionary which maps each app family to a number
        # (0 is Benign, others are malware). In the same loop create
        # a dict with the family ID for each malware (hash)
        mal_fam = dict()
        mal_fam_map = {'Benign': 0}
        with open(self.families_path, 'rb') as csvfile:
            mal_fam_reader = csv.reader(csvfile)
            next(mal_fam_reader)  # Skipping the first header line
            for row in mal_fam_reader:
                if row[1] not in mal_fam_map:
                    mal_fam_map[row[1]] = len(mal_fam_map)
                mal_fam[row[0]] = mal_fam_map[row[1]]
        # The next is the inverted dict with app family number as key
        mal_fam_map_inverted = {k[1]: k[0] for k in mal_fam_map.items()}

        # Adding the extra parameters to the dataset object
        ds.original_idx = original_idx
        ds.feat_family_idx = feat_family_idx
        ds.feat_desc = feat_desc
        ds.app_family_map = mal_fam_map_inverted
        ds.mal_family = mal_fam

        return ds

    def _get_data(self, file_url, dl_folder):
        """Download input datafile, unzip and store in output_path.

        Parameters
        ----------
        file_url : str
            URL of the file to download.
        dl_folder : str
            Path to the folder where to store the downloaded file.

        """
        user = DREBIN_USER + ':' + settings.parse_config(
            settings.SECML_CONFIG, 'drebin', 'dl_password')
        # Generate the full path to the downloaded file
        f_dl = dl_file(file_url, dl_folder, user=user, md5_digest=DREBIN_MD5)
        # Extract the content of downloaded file
        tarfile.open(name=f_dl, mode='r:gz').extractall(dl_folder)
        # Remove download file
        fm.remove_file(f_dl)
        # Now move all data to an upper folder if needed
        if not fm.file_exist(self.data_path):
            sub_d = fm.join(dl_folder, fm.listdir(dl_folder)[0])
            for e in fm.listdir(sub_d):
                e_full = fm.join(sub_d, e)  # Full path to current element
                try:  # Call copy_file or copy_folder when applicable
                    if fm.file_exist(e_full) is True:
                        fm.copy_file(e_full, dl_folder)
                    elif fm.folder_exist(e_full) is True:
                        fm.copy_folder(e_full, fm.join(dl_folder, e))
                except:
                    pass
            # Check that the main dataset file is now in the correct folder
            if not fm.file_exist(self.data_path):
                raise RuntimeError("dataset main file not available!")
            # The subdirectory can now be removed
            fm.remove_folder(sub_d, force=True)
