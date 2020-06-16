"""
.. module:: Creator
   :synopsis: Superclass and factory for all the other classes

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from importlib import import_module
from inspect import isclass, getmembers
from functools import wraps

from secml.settings import SECML_STORE_LOGS, SECML_LOGS_PATH
from secml.core.attr_utils import is_writable, is_readable, \
    extract_attr, as_public, has_protected, as_protected, get_private
from secml.core.type_utils import is_str
import secml.utils.pickle_utils as pck
from secml.utils.list_utils import find_duplicates
from secml.utils import CLog, SubLevelsDict


class CCreator:
    """The magnificent global superclass.

    Attributes
    ----------
    class_type : str
        Class type identification string. If not defined,
         class will not be instantiable using `.create()`.
    __super__ : str or None
        String with superclass name.
        Can be None to explicitly NOT support `.create()` and `.load()`.

    """
    __class_type = None  # Must be re-defined to support `.create()`
    __super__ = None  # Name of the superclass (if `.create()` or `.load()` should be available)

    # Ancestor logger, level 'WARNING' by default
    _logger = CLog(
        add_stream=True,
        file_handler=SECML_LOGS_PATH if SECML_STORE_LOGS is True else None)

    @property
    def class_type(self):
        """Defines class type."""
        try:  # Convert the private attribute to public property
            return get_private(self.__class__, 'class_type')
        except AttributeError:
            raise AttributeError("'class_type' not defined for '{:}'"
                                 "".format(self.__class__.__name__))

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

        The list of subclasses of calling superclass is looked for any class
        defining `class_item = 'value'`. If found, the class type is listed.

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
            Instance of the class having the given type (`class_type`)
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

        # Get all the subclasses of the superclass
        subclasses = cls.get_subclasses()

        # Get all class types from the list of subclasses (to check duplicates)
        class_types = import_class_types(subclasses)

        # Check for duplicates
        _check_class_types_duplicates(class_types, subclasses)

        # Everything seems fine now, look for desired class type
        for class_data in subclasses:
            if get_private(class_data[1], 'class_type', None) == class_item:
                return class_data[1](*args, **kwargs)

        raise NameError("no class of type `{:}` is a subclass of '{:}' "
                        "from module '{:}'".format(
                            class_item, cls.__name__, cls.__module__))

    @classmethod
    def get_subclasses(cls):
        """Get all the subclasses of the calling class.

        Returns
        -------
        subclasses : list of tuple
            The list containing a tuple (class.__name__, class) for
            each subclass of calling class. Keep in mind that in Python
            each class is a "subclass" of itself.

        """
        def get_subclasses(sup_cls):
            subcls_list = []
            for subclass in sup_cls.__subclasses__():
                subcls_list.append((subclass.__name__, subclass))
                subcls_list += get_subclasses(subclass)
            return subcls_list

        subclasses = get_subclasses(cls)

        # the superclass is a "subclass" of itself (in Python)
        subclasses.append((cls.__name__, cls))

        return subclasses

    @classmethod
    def list_class_types(cls):
        """This method lists all types of available subclasses of calling one.

        The list of subclasses of calling superclass is looked for any class
        defining `class_item = 'value'`. If found, the class type is listed.

        Returns
        -------
        types : list
            List of the types of available subclasses of calling class.

        """
        # Why this method is a classmethod? Just an exception for simplicity
        # classmethods should normally return an instance of calling class
        if cls.__super__ != cls.__name__:
            raise TypeError("only superclasses can be used.")

        # Get all the subclasses of the superclass
        subclasses = cls.get_subclasses()

        # Get all class types from the list of subclasses (to check duplicates)
        class_types = import_class_types(subclasses)

        # Check for duplicates
        _check_class_types_duplicates(class_types, subclasses)

        return class_types

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

        # Get all the subclasses of the superclass
        subclasses = cls.get_subclasses()

        # Look for desired class type
        for class_data in subclasses:
            if get_private(class_data[1], 'class_type', None) == class_type:
                return class_data[1]

        raise NameError("no class of type `{:}` found within the package "
                        "of class '{:}'".format(class_type, cls.__module__))

    def get_params(self):
        """Returns the dictionary of class hyperparameters.

        A hyperparameter is a PUBLIC or READ/WRITE attribute.

        """
        # We extract the PUBLIC (pub) and the READ/WRITE (rw) attributes
        # from the class dictionary, than we build a new dictionary using
        # as keys the attributes names without the accessibility prefix
        params = SubLevelsDict((as_public(k), getattr(self, as_public(k)))
                               for k in extract_attr(self, 'pub+rw'))

        # Now look for any parameter inside the accessible attributes
        for k in extract_attr(self, 'r'):
            # Extract the contained object (if any)
            k_attr = getattr(self, as_public(k))
            if hasattr(k_attr, 'get_params') and len(k_attr.get_params()) > 0:
                # as k_attr has one or more parameters, it's a parameter itself
                params[as_public(k)] = k_attr

        return params

    def set_params(self, params_dict, copy=False):
        """Set all parameters passed as a dictionary {key: value}.

        This function natively takes as input the dictionary
        created by `.get_params`.
        Only parameters, i.e. PUBLIC or READ/WRITE attributes, can be set.

        For more information on the setting behaviour see `.CCreator.set`.

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
        get_params : returns the dictionary of class parameters.

        """
        for param_name in params_dict:
            # Call single attribute set method
            self.set(param_name, params_dict[param_name], copy)

    def set(self, param_name, param_value, copy=False):
        """Set a parameter of the class.

        Only writable attributes of the class,
        i.e. PUBLIC or READ/WRITE, can be set.

        The following checks are performed before setting:
         - if `param_name` is an attribute of current class, set directly;
         - else, iterate over __dict__ and look for a class attribute
            having the desired parameter as an attribute;
         - else, if attribute is not found on the 2nd level,
            raise AttributeError.

        If possible, a reference to the attribute to set is assigned.
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
            extracted, a deepcopy of the parameter value is done first.

        """
        def copy_attr(attr_tocopy):
            from copy import deepcopy
            return deepcopy(attr_tocopy)

        # Support for recursive setting, e.g. -> kernel.gamma
        param_name = param_name.split('.')

        attr0 = param_name[0]
        if hasattr(self, attr0):
            # Level 0 set or multiple sublevels set?
            if len(param_name) == 1:  # Set attribute directly
                # Level 0 attribute must be writable
                # PUBLIC and READ/WRITE accessibility is checked
                if not is_writable(self, attr0):
                    raise AttributeError(
                        "can't set `{:}`, must be writable.".format(attr0))
                setattr(self, attr0, copy_attr(param_value)
                        if copy is True else param_value)
                return
            else:  # Start recursion on sublevels
                # Level 0 attribute must be accessible (readable)
                # PUBLIC, READ/WRITE and READ ONLY accessibility is checked
                if not is_readable(self, attr0):
                    raise AttributeError(
                        "can't set `{:}`, must be accessible.".format(attr0))
                sub_param_name = '.'.join(param_name[1:])
                # Calling `.set` method of the next sublevel
                getattr(self, attr0).set(sub_param_name, param_value, copy)
                return

        # OLD STYLE SET: recursion on 2 levels only to set a subattribute
        # The first subattribute found is set...
        else:
            # Look for the attribute inside all class attributes
            for attr_name in self.__dict__:
                # Extract the current attribute
                attr = getattr(self, attr_name)
                # If parameter is an attribute of current attribute set it
                if hasattr(attr, attr0):
                    # Attributes to set must be writable
                    # PUBLIC and READ/WRITE accessibility is checked
                    if not is_writable(attr, attr0):
                        raise AttributeError(
                            "can't set `{:}`, must be writable.".format(attr0))
                    setattr(attr, attr0, copy_attr(param_value)
                            if copy is True else param_value)
                    return

        # Attribute not found, raise AttributeError
        raise AttributeError(
            "'{:}', or any of its attributes, has attribute '{:}'"
            "".format(self.__class__.__name__, attr0))

    def get_state(self):
        """Returns the object state dictionary.

        Returns
        -------
        dict
            Dictionary containing the state of the object.

        """
        # We extract the PUBLIC (pub), READ/WRITE (rw) and READ ONLY (r)
        # attributes from the class dictionary, than we build a new dictionary
        # using as keys the attributes names without the accessibility prefix
        state = dict((as_public(k), getattr(self, as_public(k)))
                     for k in extract_attr(self, 'pub+rw+r'))

        # Get the state of the deeper objects
        # Use list(state) as state size will change during iteration
        for attr in list(state):
            if isinstance(state[attr], CCreator):
                state_deep = state[attr].get_state()
                # Replace `attr` with its attributes's state
                for attr_deep in state_deep:
                    attr_full_key = attr + '.' + attr_deep
                    state[attr_full_key] = state_deep[attr_deep]
                del state[attr]

        return dict(state)

    def set_state(self, state_dict, copy=False):
        """Sets the object state using input dictionary.

        Only readable attributes of the class,
        i.e. PUBLIC or READ/WRITE or READ ONLY, can be set.

        If possible, a reference to the attribute to set is assigned.
        Use `copy=True` to always make a deepcopy before set.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state of the object.
        copy : bool, optional
            By default (False) a reference to the attribute to
            assign is set. If True or a reference cannot be
            extracted, a deepcopy of the attribute is done first.

        """
        def copy_attr(attr_tocopy):
            from copy import deepcopy
            return deepcopy(attr_tocopy)

        for param_name in state_dict:

            # Extract the value of the attribute to set
            param_value = state_dict[param_name]

            # Support for recursive setting, e.g. -> kernel.gamma
            param_name = param_name.split('.', 1)

            # Attributes to set in this function must be readable
            # PUBLIC, READ/WRITE and READ ONLY accessibility is checked
            if not is_readable(self, param_name[0]):
                raise AttributeError(
                    "can't set `{:}`, must be readable.".format(param_name[0]))

            attr0 = param_name[0]
            if hasattr(self, attr0):
                # 1 level set or multiple sublevels set?
                if len(param_name) == 1:  # Set attribute directly
                    # If writable (public or property with setter)
                    if is_writable(self, attr0):  # Use main `.set`
                        self.set(attr0, param_value, copy=copy)
                        continue  # Attribute set, go to next one
                    else:  # Maybe is read-only (property with only getter)?
                        # If exists, set the protected attribute
                        if has_protected(self, attr0):
                            attr0 = as_protected(attr0)
                        setattr(self, attr0, copy_attr(param_value)
                                if copy is True else param_value)
                        continue  # Attribute set, go to next one
                else:  # Start recursion on sublevels
                    # Call `.set_state` for the next level of current attribute
                    getattr(self, attr0).set_state(
                        {param_name[1]: param_value}, copy)
                    continue  # Attribute set, go to next one

            # Attribute not found, raise AttributeError
            raise AttributeError(
                "'{:}', or any of its attributes, has attribute '{:}'"
                "".format(self.__class__.__name__, attr0))

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
        """Save class object to file.

        This function stores an object to file (with pickle).

        `.load()` can be used to restore the object later.

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
        """Loads object from file.

        This function loads an object from file (with pickle).

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

    def save_state(self, path):
        """Store the object state to file.

        Parameters
        ----------
        path : str
            Path of the file where to store object state.

        Returns
        -------
        str
            The full path of the stored object.

        See Also
        --------
        get_state : Returns the object state dictionary.

        """
        return pck.save(path, self.get_state())

    def load_state(self, path):
        """Sets the object state from file.

        Parameters
        ----------
        path : str
            The full path of the file from which to load the object state.

        See Also
        --------
        set_state : Sets the object state using input dictionary.

        """
        # Copy not needed for objects loaded from disk
        self.set_state(pck.load(path), copy=False)

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


def import_class_types(classes):
    """Returns types associated with input list of classes.

    Abstract properties are ignored.

    Returns
    -------
    types : list
        List of class types associated with input list of classes.

    """
    # Get all class types from the input list of classes (to check duplicates)
    # Leaving out the classes not defining a class_type
    class_types = map(
        lambda class_file: get_private(class_file[1], 'class_type', None),
        classes)
    # skipping non string class_types -> classes not supporting creator
    return [class_type for class_type in
            class_types if isinstance(class_type, str)]


def _check_class_types_duplicates(class_types, classes):
    """Check duplicated types for input list of class types."""
    duplicates = find_duplicates(class_types)
    if len(duplicates) != 0:  # Return the list of classes with duplicate type
        duplicates_classes = [
            (class_tuple[0], get_private(class_tuple[1], 'class_type'))
            for class_tuple in classes if
            get_private(class_tuple[1], 'class_type', None) in duplicates]
        raise ValueError("following classes have the same class type. Fix "
                         "before continue. {:}".format(duplicates_classes))
