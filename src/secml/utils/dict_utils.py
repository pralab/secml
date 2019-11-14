"""
.. module:: DictionaryUtils
   :synopsis: Collection of mixed utilities for Dictionaries

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from collections.abc import MutableMapping

__all__ = ['load_dict', 'merge_dicts', 'invert_dict',
           'LastInDict', 'SubLevelsDict']


def load_dict(file_path, values_dtype=str, encoding='ascii'):
    """Load dictionary from textfile.

    Each file's line should be <key: value>

    Parameters
    ----------
    file_path : str
        Full path to the file to read.
    values_dtype : dtype
        Datatype of the values. Default str (string).
    encoding : str, optional
        Encoding to use for reading the file. Default 'ascii'.

    Returns
    -------
    dictionary : dict
        Loaded dictionary with one key for each
        line in the input text file.

    """
    new_dict = {}
    with open(file_path, mode='rt', encoding=encoding) as df:
        for key_line in df:
            # a line is 'key: value'
            key_line_split = key_line.split(':')
            try:
                # Removing any space from key value before setting
                new_dict[key_line_split[0]] = values_dtype(key_line_split[1].strip())
            except IndexError:
                raise ValueError("line '{:}' is not valid.".format(key_line))
    return new_dict


def merge_dicts(*dicts):
    """Shallow copy and merge any number of input dicts.

    Precedence goes to key value pairs in latter dicts.

    Parameters
    ----------
    dicts : dict1, dict2, ...
        Any sequence of dict objects to merge.

    Examples
    --------
    >>> from secml.utils import merge_dicts

    >>> d1 = {'attr1': 100, 'attr2': 200}
    >>> d2 = {'attr3': 300, 'attr1': 999}  # Redefining `attr1`

    >>> merge_dicts(d1, d2)  # Value of `attr1` will be set according to `d2` dictionary
    {'attr3': 300, 'attr2': 200, 'attr1': 999}

    """
    result = {}
    for dict_i in dicts:
        result.update(dict_i)
    return result


def invert_dict(d):
    """Returns a new dict with keys as values and values as keys.

    Parameters
    ----------
    d : dict
        Input dictionary. If one value of the dictionary is a list or a tuple,
        each element of the sequence will be considered separately.

    Returns
    -------
    dict
        The new dictionary with d keys as values and d values as keys.
        In the case of duplicated d values, the value of the resulting key
        of the new dictionary will be a list with all the corresponding d keys.

    Examples
    --------
    >>> from secml.utils.dict_utils import invert_dict

    >>> a = {'k1': 2, 'k2': 2, 'k3': 1}
    >>> print(invert_dict(a))
    {1: 'k3', 2: ['k1', 'k2']}

    >>> a = {'k1': 2, 'k2': [2,3,1], 'k3': 1}
    >>> print(invert_dict(a))
    {1: ['k2', 'k3'], 2: ['k1', 'k2'], 3: 'k2'}

    """
    def tolist(x): return [x] if not isinstance(x, (list, tuple)) else list(x)
    new_d = {}
    for k in d.items():
        for v in tolist(k[1]):
            i = k[0]
            if v in new_d:
                # If the key has already been set create a list for the values
                i = tolist(i)
                i = tolist(new_d[v]) + i
            new_d[v] = i
    return new_d


class LastInDict(MutableMapping):
    """Last In Dictionary.

    A standard dictionary that keeps in memory the key of the last set item.
    The setting behaviour is queue-like: a single element can be inserted
    in the dictionary each time.

    The last key can be changes manually calling `LastInDict.lastitem_id = key`.

    Examples
    --------
    >>> from secml.utils import LastInDict

    >>> li = LastInDict()

    >>> li['key1'] = 123
    >>> li['key2'] = 102030

    >>> li.lastin_key
    'key2'
    >>> li.lastin
    102030

    """
    def __init__(self):
        self._data = dict()
        self._rw_lastin_key = None

    @property
    def lastin(self):
        return self._data[self.lastin_key]

    @property
    def lastin_key(self):
        return self._rw_lastin_key

    @lastin_key.setter
    def lastin_key(self, key):
        if key not in self._data:
            raise KeyError("unknown key '{:}'.".format(key))
        self._rw_lastin_key = key

    def __setitem__(self, key, value):
        self._data[key] = value
        self.lastin_key = key

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        self.lastin_key = None if self.lastin_key == key else self.lastin_key
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for key in self._data:
            yield key


class SubLevelsDict(MutableMapping):
    """Sub-Levels Dictionary.

    A standard dictionary that allows easy access to attributes of
    contained objects at infinite deep.

    Examples
    --------
    >>> from secml.utils import SubLevelsDict

    >>> class Foo:
    ...     attr2 = 5

    >>> li = SubLevelsDict({'attr1': Foo()})

    >>> print(type(li['attr1']))
    <class 'dict_utils.Foo'>
    >>> print(li['attr1.attr2'])
    5

    >>> li['attr1.attr2'] = 10  # Subattributes can be set in the same way
    >>> print(li['attr1.attr2'])
    10

    """
    def __init__(self, data):
        self._data = dict(data)

    def __setitem__(self, key, value):
        # Support for recursion, e.g. -> attr1.attr2
        key = key.split('.')

        # Setting a key element works like in dictionaries
        if len(key) == 1:
            self._data[key[0]] = value
            return

        # The first element of key is a key of the dictionary
        data = self._data[key[0]]
        # Now get the desired subattributes recursively,
        # until the level before the last is reached
        for key_split in key[1:-1]:
            data = getattr(data, key_split)

        # The last subattribute must be an attribute of the deepest level
        if hasattr(data, key[-1]):
            setattr(data, key[-1], value)
        else:
            raise AttributeError("'{:}' not found.".format('.'.join(key)))

    def __getitem__(self, key):
        # Support for recursion, e.g. -> attr1.attr2
        key = key.split('.')
        # The first element of key is a key of the dictionary
        data = self._data[key[0]]
        # Now get the desired subattributes recursively,
        # until the last level is reached
        for key_split in key[1:]:
            data = getattr(data, key_split)

        return data

    def __delitem__(self, key):
        if len(key.split('.')) != 1:
            raise ValueError("only first-level attributes can be removed.")
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        # Support for recursion, e.g. -> attr1.attr2
        key = key.split('.')

        # Check the first element, is a key of the dictionary
        if key[0] not in self._data:
            return False

        # The first element of key is a key of the dictionary
        data = self._data[key[0]]
        # Now get the desired subattributes recursively
        for key_split in key[1:]:
            if not hasattr(data, key_split):
                return False
            data = getattr(data, key_split)
        return True

    def __iter__(self):
        for key in self._data:
            yield key

    def __repr__(self):
        return dict.__repr__(self._data)
