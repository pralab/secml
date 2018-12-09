"""
.. module:: ClassCreator
   :synopsis: Creates an instance of a supported class in a package

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Igino Corona <igino.corona@diee.unica.it>
.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>

"""
from importlib import import_module
from inspect import isclass, getmembers
from functools import wraps

from secml.core.attr_utils import is_public, extract_attr, \
    as_public, as_private, get_private
from secml.core.type_utils import is_str
import secml.utils.pickle_utils as pck
from secml.utils.list_utils import find_duplicates
from secml.utils import CLog, SubLevelsDict


class CCreator(object):
    """The magnificent global superclass.

    Attributes
    ----------
    class_type : str or None
        Class type or identification string.
        Can be None to explicitly NOT support `.create()`.
    __super__ : str or None
        String with superclass name.
        Can be None to explicitly NOT support `.create()` and `.load()`.

    """
    __class_type = None  # This attribute must be re-defined to support `.create()`
    __super__ = None  # Leaving this None will make `create` and `load` not supported

    # TODO: MAKE FILE PATH/NAME DYNAMIC
    _logger = CLog(add_stream=True, file_handler='logs.log')  # Ancestor logger, level 'WARNING' by default

    @property
    def class_type(self):
        """Defines class type."""
        return get_private(self.__class__, 'class_type', None)

    @property
    def logger(self):
        """Logger for current object."""
        return self._logger.get_child(self.__class__.__name__ + '.' + str(hex(id(self))))

    @property
    def verbose(self):
        """Verbosity level of logger output.

        Available levels are:
         0 = no verbose output
         1 = info-level logging
         2 = debug-level logging

        """
        verbosity_lvls = {30: 0, 20: 1, 10: 2}  # 30: WARNING, 20: INFO, 10: DEBUG
        return verbosity_lvls[self.logger.level]

    @verbose.setter
    def verbose(self, level):
        """Sets verbosity level of logger."""
        verbosity_lvls = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}
        if level not in verbosity_lvls:
            raise ValueError("Verbosity level {:} not supported.".format(level))
        self.logger.set_level(verbosity_lvls[level])

    @staticmethod
    def timed(msg=None):
        """Timer decorator.

        Returns a decorator that can be used to measure
        execution time of any method.
        Performance data will be stored inside the class logger.
        Messages will be logged using the INFO logging level.
        As this decorator accepts optional arguments,
        must be called as a method. See examples.

        Parameters
        ----------
        msg : str or None, optional
            Custom message to display when entering the timed block.
            If None, "Entering timed block `method_name`..." will printed.

        """
        def wrapper(fun):
            @wraps(fun)  # To make wrapped_fun work as fun_timed
            def wrapped_fun(self, *args, **kwargs):
                # Wrap fun again and add the timed decorator
                @self.logger.timed(msg=msg)
                @wraps(fun)  # To make fun_timed work as fun
                def fun_timed(*fun_args, **fun_wargs):
                    return fun(*fun_args, **fun_wargs)
                return fun_timed(self, *args, **kwargs)
            return wrapped_fun
        return wrapper

    @classmethod
    def create(cls, class_item=None, *args, **kwargs):
        """This method creates an instance of a class with given type.

        Calling superclass's package is looked for any subclass defining
        `class_item = 'value'`. If found, the class is instantiated and a
        reference is returned.

        Also a class instance can be passed as main argument.
        In this case the class instance is returned as is.

        Parameters
        ----------
        class_item : str or class instance or None, optional
            Type of the class to instantiate.
            If a class instance of cls is passed, instead, it returns
             the instance directly.
            If this is None, an instance of the classing superclass is created.
        args, kwargs : optional arguments
            Any other argument for the class to create.
            If a class instance is passed as `class_item`,
            optional arguments are NOT allowed.

        Returns
        -------
        instance_class : any class
            Instance of the class having the given type (class_type)
            or the same class instance passed as input.

        """
        if cls.__super__ != cls.__name__:
            raise TypeError("classes can be created from superclasses only.")

        # Create an instance of the calling superclass
        if class_item is None:
            return cls(*args, **kwargs)  # Pycharm says are unexpected args

        # We accept strings and class instances only
        if isclass(class_item):  # Returns false for instances
            raise TypeError("creator only accepts a class type "
                            "as string or a class instance.")

        # CCreator cannot be created!
        if class_item.__class__ == CCreator:
            raise TypeError("class 'CCreator' is not callable.")

        # If a class instance is passed, it's returned as is
        if not is_str(class_item):
            if not isinstance(class_item, cls):
                raise TypeError("input instance should be a {:} "
                                "subclass.".format(cls.__name__))
            if len(args) + len(kwargs) != 0:
                raise TypeError("optional arguments are not allowed "
                                "when a class instance is passed.")
            return class_item

        # Get all the classes from the package
        package_classes = import_package_classes(cls)

        # Get all class types from the package (to check duplicates)
        package_types = import_package_types(package_classes)

        # Check for duplicates
        _check_package_types_duplicates(package_classes, package_types)

        # Everything seems fine now, look for desired class type
        for class_data in package_classes:
            if get_private(class_data[1], 'class_type', None) == class_item:
                return class_data[1](*args, **kwargs)

        raise NameError("no class of type `{:}` found within the package "
                        "of class '{:}'".format(class_item, cls.__module__))

    @classmethod
    def list_class_types(cls):
        """This method lists all types of available subclasses of calling one.

        Calling superclass's package is looked for any subclass defining
        `class_item = 'value'`. If found, the class type is listed.

        Returns
        -------
        types : list
            List of the types of available subclasses of calling class.

        """
        # Why this method is a classmethod? Just an exception for simplicity
        # classmethods should normally return an instance of calling class
        if cls.__super__ != cls.__name__:
            raise TypeError("only superclasses can be used.")

        # Get all the classes from the package
        package_classes = import_package_classes(cls)

        # Get all class types from the package (to check duplicates)
        package_types = import_package_types(package_classes)

        # Check for duplicates
        _check_package_types_duplicates(package_classes, package_types)

        return package_types

    @classmethod
    def get_class_from_type(cls, class_type):
        """Return the class associated with input type.

        This will NOT check for classes with duplicated class type.
        The first class found with matching type will be returned.

        Parameters
        ----------
        class_type : str
            Type of the class which will be looked up for.

        Returns
        -------
        class_obj : class
            Desired class, if found. This is NOT an instance of the class.

        """
        # Why this method is a classmethod? Just an exception for simplicity
        # classmethods should normally return an instance of calling class
        if cls.__super__ != cls.__name__:
            raise TypeError("only superclasses can be used.")

        # Get all the classes from the package
        package_classes = import_package_classes(cls)

        # Look for desired class type
        for class_data in package_classes:
            if get_private(class_data[1], 'class_type', None) == class_type:
                return class_data[1]

        raise NameError("no class of type `{:}` found within the package "
                        "of class '{:}'".format(class_type, cls.__module__))

    def clear(self):
        """Resets internal attributes of all the hierarchy.

        To properly support the clear framework, define a `__clear` method
        that executes the desired operations.

        """
        if self.is_clear() is True:  # If instance is clear, no need to clear
            return
        # __mro__ returns the class hierarchy (reverse order)
        for base in self.__class__.__mro__:
            # `__clear` is called only if defined
            if hasattr(base, as_private(base, 'clear')):
                getattr(self, as_private(base, 'clear'))()

    def is_clear(self):
        """Returns True if object is clear.

        This function returns True if the internal attributes of
        the instance are cleared, namely, if the instance has
        not performed any operation after init.

        Notes
        -----
        Override this method to correctly support the `clear` framework.
        Otherwise, the class will be always considered NOT cleared and
        thus the clear() method will be called.

        """
        return False

    def get_params(self):
        """Returns the dictionary of class parameters.

        A parameter is a PUBLIC or READ/WRITE attribute.

        """
        # We extract the PUBLIC (pub) and the READ/WRITE (rw) attributes
        # from the class dictionary, than we build a new dictionary using
        # as keys the attributes names without the accessibility prefix
        return SubLevelsDict((as_public(k), getattr(self, as_public(k)))
                             for k in extract_attr(self, 'pub+rw'))

    def set_params(self, params_dict, copy=False):
        """Set all parameters passed as a dictionary {key: value}.

        This function natively takes as input the dictionary
        created by `.get_params()`.
        Only parameters, i.e. PUBLIC or READ/WRITE attributes, can be set.
        RW parameters must be set using their real name, e.g. use
        `attr` instead of `_rw_attr`.

        For more informations on the setting behaviour see `.CCreator.set`.

        If possible, a reference to the parameter to set is assigned.
        Use `copy=True` to always make a deepcopy before set.

        Parameters
        ----------
        params_dict : dict
            Dictionary of parameters to set.
        copy : bool
            By default (False) a reference to the parameter to
            assign is set. If True or a reference cannot be
            extracted, a deepcopy of the parameter is done first.

        See Also
        --------
        .get_params() : returns the dictionary of class parameters.

        """
        for param_name in params_dict:
            # Call single attribute set method
            self.set(param_name, params_dict[param_name], copy)

    def set(self, param_name, param_value, copy=False):
        """Set a parameter that has a specific name to a specific value.

        Only parameters, i.e. PUBLIC or READ/WRITE attributes, can be set.
        RW parameters must be set using their real name, e.g. use
        `attr` instead of `_rw_attr`.

        The following checks are performed before setting:
         - if parameter is an attribute of current class, set directly;
         - else, iterate over __dict__ and look for a class attribute
            having the desired parameter as an attribute;
         - else, if attribute is not found on the 2nd level,
            raise AttributeError.

        If possible, a reference to the parameter to set is assigned.
        Use `copy=True` to always make a deepcopy before set.

        Parameters
        ----------
        param_name : str
            Name of the parameter to set.
        param_value : any
            Value to set for the parameter.
        copy : bool
            By default (False) a reference to the parameter to
            assign is set. If True or a reference cannot be
            extracted, a deepcopy of the parameter is done first.

        """
        def copy_attr(attr_tocopy):
            from copy import deepcopy
            return deepcopy(attr_tocopy)

        # Support for recursive setting, e.g. -> kernel.gamma
        param_name = param_name.split('.')

        # Parameters settable in this function must be public.
        # READ/WRITE accessibility is then checked by the setter...
        if not is_public(self, param_name[0]):
            raise AttributeError(
                "can't set `{:}`, must be public.".format(param_name[0]))

        if hasattr(self, param_name[0]):
            # 1 level set or multiple sublevels set?
            if len(param_name) == 1:  # Set parameter directly
                setattr(self, param_name[0], copy_attr(
                    param_value) if copy is True else param_value)
                return
            else:  # Start recursion on sublevels
                sub_param_name = '.'.join(param_name[1:])
                # Calling `.set` method of the next sublevel
                getattr(self, param_name[0]).set(
                    sub_param_name, param_value, copy)
                return

        # OLD STYLE SET: recursion on 2 levels only to set a subattribute
        # The first subattribute found is set...
        else:
            # Look for parameter inside all class attributes
            for attr_name in self.__dict__:
                # Extract the current attribute
                attr = getattr(self, attr_name)
                # If parameter is an attribute of current attribute set it
                if hasattr(attr, param_name[0]):
                    setattr(attr, param_name[0], copy_attr(
                        param_value) if copy is True else param_value)
                    return

        # If we haven't found desired parameter anywhere, raise AttributeError
        raise AttributeError("'{:}', or any of its attributes, has "
                             "parameter '{:}'".format(
                                 self.__class__.__name__, param_name))

    def copy(self):
        """Returns a shallow copy of current class.

        As shallow copy creates a new instance of current object and
        then insert in the new object a reference (if possible) to
        each attribute of the original object.

        """
        from copy import copy
        return copy(self)

    def __copy__(self, *args, **kwargs):
        """Called when copy.copy(object) is called."""
        from copy import copy
        new_obj = self.__new__(self.__class__)
        for attr in self.__dict__:
            new_obj.__dict__[attr] = copy(self.__dict__[attr])
        return new_obj

    def deepcopy(self):
        """Returns a deep copy of current class.

        As deep copy is time consuming in most cases, can sometimes
        be acceptable to select a subset of attributes and assign
        them to a new instance of the current class using `.set_params`.

        """
        from copy import deepcopy
        return deepcopy(self)

    def __deepcopy__(self, memo, *args, **kwargs):
        """Called when copy.deepcopy(object) is called.

        `memo` is a memory dictionary needed by `copy.deepcopy`.

        """
        from copy import deepcopy
        new_obj = self.__new__(self.__class__)
        for attr in self.__dict__:
            new_obj.__dict__[attr] = deepcopy(self.__dict__[attr], memo)
        return new_obj

    def save(self, path):
        """Save class object using pickle.

        Store the current class instance to disk, preserving
        the state of each attribute.

        `.load()` can be used to restore the instance later.

        Parameters
        ----------
        path : str
            Path of the target object file.

        Returns
        -------
        obj_path : str
            The full path of the stored object.

        """
        return pck.save(path, self)

    @classmethod
    def load(cls, path):
        """Loads class from pickle object.

        This function loads any object stored with pickle
        or cPickle and any output of `.save()`.

        The object can be correctly loaded in the following cases:
         - loaded and calling class have the same type.
         - calling class is the superclass of the loaded class's package.
         - calling class is `.CCreator`.

        Parameters
        ----------
        path : str
            Path of the target object file.

        """
        loaded_obj = pck.load(path)
        if loaded_obj.__class__ == cls or cls == CCreator or \
                (has_super(loaded_obj) and cls.__name__ == loaded_obj.__super__):
            return loaded_obj
        else:
            err_str = "'{0}' can be loaded from: '{0}'".format(
                loaded_obj.__class__.__name__)
            if has_super(loaded_obj):
                err_str += ", '{:}'".format(loaded_obj.__super__)
            raise TypeError(err_str + " or 'CCreator'.")

    def __repr__(self):
        """Defines print behaviour."""
        out_repr = self.__class__.__name__ + "{"
        for k in extract_attr(self, 'pub+rw+r'):
            pub_attr_name = as_public(k)
            out_repr += "'{:}': ".format(pub_attr_name)
            out_repr += repr(getattr(self, pub_attr_name))
            out_repr += ", "
        return out_repr.rstrip(', ') + "}"


def has_super(cls):
    """Returns True if input class `__super__` is not None.

    `__super__` is defined and not None for class trees having
    a main superclass and one or more inherited classes.

    Parameters
    ----------
    cls : obj
        Any class or class isntance.

    """
    return hasattr(cls, '__super__') and cls.__super__ is not None


def import_package_classes(cls):
    """Get all the classes inside a package.

    Returns
    -------
    members : list
        Return all members of an object as (name, value)
        pairs sorted by name.

    """
    # Get all modules inside the package of calling superclass
    package_name = cls.__module__
    # Leaving out the last part of __module__ string as is `cls` filename
    # But only if module is not the main (a single file)
    if package_name != '__main__':
        package_name = package_name.rpartition('.')[0]
    # Import the entire package
    package = import_module(package_name)
    # Get the classes only from the package
    return getmembers(package, isclass)


def import_package_types(package_classes):
    """Returns types associated with input list of package classes.

    Abstract properties are ignored.

    Returns
    -------
    types : list
        List of class types associated with input list of package classes.

    """
    # Get all class types from the package (to check duplicates)
    # Leaving out the classes not defining a class_type
    package_types = map(
        lambda class_file: get_private(class_file[1], 'class_type', None),
        package_classes)
    # skipping non string class_types -> classes not supporting creator
    return [class_type for class_type in
            package_types if isinstance(class_type, str)]


def _check_package_types_duplicates(package_classes, package_types):
    """Check duplicated types for input list of package classes."""
    # Check for duplicates
    duplicates = find_duplicates(package_types)
    if len(duplicates) != 0:
        duplicates_classes = [
            (class_tuple[0], get_private(class_tuple[1], 'class_type'))
            for class_tuple in package_classes if
            get_private(class_tuple[1], 'class_type', None) in duplicates]
        raise ValueError("following classes have the same class type. Fix "
                         "before continue. {:}".format(duplicates_classes))
