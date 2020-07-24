"""
.. module:: Logger
   :synopsis: Log and store code information on disk.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import logging
import time
import sys
import os
import warnings
from functools import wraps

# Custom logging level that DISABLE logging of all messages
DISABLE = 100
logging.addLevelName(100, 'DISABLE')
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

# Enabling capture of warnings
logging.captureWarnings(True)

# Default formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class CLog:
    """Manager for logging and logfiles.

    Logger can be used to save important runtime code information
    to disk instead of built-in function 'print'. Along with any
    print-like formatted string, the logger stores full time stamp
    and calling class name.

    The default filename of a log file is ``logs.log``. This will be
    placed in the same directory of the calling file.

    Logging levels currently available and target purpose:
     * DISABLE - 100: disable all logging.
     * CRITICAL - 50: critical error messages.
     * ERROR - 40: standard error messages.
     * WARNING - 30: standard error messages.
     * INFO - 20: general info logging.
     * DEBUG - 10: debug logging only.

    Logger is fully integrated to the :class:`.CTimer` class in order to log
    performance of a desired method or routine.

    Parameters
    ----------
    level : LOG_LEVEL, int or None, optional
        Initial logging level. Default is None, meaning that the current
        logging level will be preserved if the logger has already been created.
    logger_id : str or None, optional
        Identifier of the logger. Default None.
        If None, creates a logger which is the root of the hierarchy
    add_stream : bool, optional
        If True, attach a stream handler to the logger. Default True.
        A stream handler prints to stdout the logged messages.
    file_handler : str or None, optional
        If a string, attach a file handler to the logger. Default None.
        A file handler stores to the specified path the logged messages.
    propagate : bool, optional
        If True, messages logged to this logger will be passed to the
        handlers of higher level (ancestor) loggers, in addition to any
        handler attached to this logger. Default False.

    Notes
    -----
    Unlike most of the Python logging modules, our implementation can be
    fully used inside parallelized code.

    See Also
    --------
    .CTimer : Manages performance monitoring and logging.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.utils import CLog

    >>> log = CLog().warning("{:}".format(CArray([1,2,3])))  # doctest: +SKIP
    ... - WARNING - CArray([1 2 3])

    """

    def __init__(self, level=None, logger_id=None, add_stream=True,
                 file_handler=None, propagate=False):
        # Setting up logger with default logging level (WARNING)
        self._logger_id = None if logger_id is None else str(logger_id)
        self._propagate = propagate
        self._logger = None
        self._set_logger()  # Calls getLogger(logger_id)
        if add_stream is True:  # Attach a stream handler
            self.attach_stream()
        if file_handler is not None:  # Attach a file handler
            self.attach_file(file_handler)
        if level is not None:
            self.set_level(level)  # Setting initial logging level

    @property
    def logger_id(self):
        """Return identifier of the logger."""
        return self._logger.name

    @property
    def level(self):
        """Return logging level."""
        return self._logger.getEffectiveLevel()

    @property
    def propagate(self):
        """If True, events logged will be passed to the handlers
        of higher level (ancestor) loggers."""
        return self._logger.propagate

    def __getstate__(self):
        """Return CLog instance before pickling."""
        state = dict(self.__dict__)
        # We now remove the store logger (will be restored after)
        del state['_logger']
        return state

    def __setstate__(self, state):
        """Reset CLog instance after pickling."""
        self.__dict__.update(state)
        # We now reinitialize logger
        self._set_logger()

    def _set_logger(self):
        """Prepares the logger."""
        self._logger = logging.getLogger(self._logger_id)
        self._logger.propagate = self._propagate

    def set_level(self, level):
        """Sets logging level of the logger."""
        self._logger.setLevel(level)

    def _add_handler(self, handler):
        """Adds handler and specifies a formatter."""
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def attach_stream(self):
        """Adds a stream handler to the logger."""
        # Handler will be attached only if not already there
        for h in self._logger.handlers:
            if isinstance(h, logging.StreamHandler):
                return
        handler = logging.StreamHandler(sys.stdout)
        self._add_handler(handler)

    def attach_file(self, filepath):
        """Adds a file handler to the logger."""
        # Handler will be attached only if not already there
        for h in self._logger.handlers:
            if isinstance(h, logging.FileHandler) and \
                    h.baseFilename == os.path.abspath(filepath):
                return
        handler = logging.FileHandler(filepath)
        self._add_handler(handler)

    def _remove_handler(self, handler):
        """Removes input handler from logger."""
        self._logger.removeHandler(handler)

    def remove_handler_stream(self):
        """Removes the stream handler from the logger."""
        handler = logging.StreamHandler(sys.stdout)
        self._remove_handler(handler)

    def remove_handler_file(self, filepath):
        """Removes the file handler from the logger."""
        handler = logging.FileHandler(filepath)
        self._remove_handler(handler)

    def get_child(self, name):
        """Return a child logger associated with ancestor.

        Parameters
        ----------
        name : str-like
            Identifier of the child logger. Can be any object
            safely convertible to string (int, float, etc.)

        Returns
        -------
        child_logger : logger
            Instance of the child logger.

        """
        # Root logger can be created using '' (empty string) as name
        parent_id = '' if self._logger_id is None else self.logger_id + '.'
        # Stream and/or file handler are set for
        # ancestors only (to avoid output duplication)
        return self.__class__(logger_id=parent_id + str(name),
                              add_stream=False, file_handler=None,
                              propagate=True)  # This is a child, so propagate

    def log(self, level, msg, *args, **kwargs):
        """Logs a message with specified level on this logger.

        The msg is the message format string, and the args are the arguments
        which are merged into msg using the string formatting operator.

        There are two keyword arguments in kwargs which are inspected: exc_info
        which, if it does not evaluate as false, causes exception information
        to be added to the logging message.
        If an exception tuple (in the format returned by sys.exc_info())
        is provided, it is used; otherwise, sys.exc_info() is called to
        get the exception information.

        The second keyword argument is extra which can be used to pass
        a dictionary  which is used to populate the __dict__ of the LogRecord
        created for the logging event with user-defined attributes.

        """
        self._logger.log(level, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Logs a message with level CRITICAl on this logger.

        See `CLog.log` for details on args and kwargs.

        """
        self._logger.critical(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Logs a message with level ERROR on this logger.

        See `CLog.log` for details on args and kwargs.

        """
        self._logger.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Logs a message with level WARNING on this logger.

        See `CLog.log` for details on args and kwargs.

        """
        self._logger.warning(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Logs a message with level INFO on this logger.

        See `CLog.log` for details on args and kwargs.

        """
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Logs a message with level DEBUG on this logger.

        See `CLog.log` for details on args and kwargs.

        """
        self._logger.debug(msg, *args, **kwargs)

    def timer(self, msg=None):
        """Starts a timed codeblock.

        Returns an instance of context manager :class:`.CTimer`.
        Performance data will be stored inside the calling logger.
        Messages will be logged using the DEBUG logging level.

        Parameters
        ----------
        msg : str or None, optional
            Custom message to display when entering the timed block.
            If None, "Entering timed block..." will printed.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.utils import CLog

        >>> log = CLog()
        >>> log.set_level(10)
        >>> with log.timer("Timing the instruction..."):
        ...     a = CArray([1,2,3])  # doctest: +ELLIPSIS
        2... - root - DEBUG - Timing the instruction...
        2... - root - DEBUG - Elapsed time: ... ms

        """
        return CTimer(log=self, msg=msg)

    def timed(self, msg=None):
        """Timer decorator.

        Returns a decorator that can be used to measure
        execution time of any method.
        Performance data will be stored inside the calling logger.
        Messages will be logged using the DEBUG logging level.
        As this decorator accepts optional arguments,
        must be called as a method. See examples.

        Parameters
        ----------
        msg : str or None, optional
            Custom message to display when entering the timed block.
            If None, "Entering timed block `function_name`..." will printed.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.utils import CLog

        >>> log = CLog()
        >>> log.set_level(10)
        >>> @log.timed()
        ... def abc():
        ...     print("Hello world!")

        >>> abc()  # doctest: +ELLIPSIS
        Hello world!

        """
        return CTimer.timed(log=self, msg=msg)

    @staticmethod
    def catch_warnings(record=False):
        """A context manager that copies and restores the warnings filter upon
        exiting the context.

        Wrapper of `warnings.catch_warnings`.

        Parameters
        ----------
        record : bool, optional
            If False (the default), the context manager returns None on entry.
            If True, a list is returned that is progressively populated with
            warning objects as seen by the context manager.

        """
        return warnings.catch_warnings(record=record)

    @staticmethod
    def filterwarnings(action, message="", category=Warning,
                       module="", lineno=0, append=False):
        """Insert an entry into the list of warnings filters (at the front).

        Wrapper of `warnings.filterwarnings`.

        Parameters
        ----------
        action : str
            One of "error", "ignore", "always", "default", "module", or "once".
        message : str, optional
            A regex that the warning message must match.
        category : class, optional
            A class that the warning must be a subclass of. Default `Warning`.
        module : str, optional
            A regex that the module name must match.
        lineno : int, optional
            An integer line number, 0 (default) matches all warnings.
        append : bool, optional
            If true, append to the list of filters.

        """
        return warnings.filterwarnings(
            action, message=message, category=category,
            module=module, lineno=lineno, append=append)


class CTimer:
    """Context manager for performance logging

    The code inside the specific context will be timed and
    performance data printed and/or logged.

    This class fully integrates with :class:`.CLog` in order to
    store to disk performance data. When no logger is specified,
    data is printed on the console output.

    Times are always stored in milliseconds (ms).

    Parameters
    ----------
    log : CLog or None, optional
        Instance of :class:`.CLog` class to be used as
        performance logger. If a logger is specified,
        timer data will not be printed on console.
    msg : str or None, optional
        Custom message to display when entering the timed block.
        If None, "Entering timed block `function_name`..." will printed.

    See Also
    --------
    .CLog : CLog and store runtime information on disk.

    Examples
    --------
    >>> from secml.array import CArray
    >>> from secml.utils import CTimer

    >>> with CTimer() as t:
    ...     a = CArray([1,2,3])  # doctest: +ELLIPSIS
    Entering timed block...
    Elapsed time: ... ms

    >>> with CTimer(msg="Timing the instruction...") as t:
    ...     a = CArray([1,2,3])  # doctest: +ELLIPSIS
    Timing the instruction...
    Elapsed time: ... ms

    >>> from secml.utils import CLog
    >>> logger = CLog()
    >>> logger.set_level(10)
    >>> with CTimer(logger) as t:
    ...     a = CArray([1,2,3])  # doctest: +ELLIPSIS
    2... - root - DEBUG - Entering timed block...
    2... - root - DEBUG - Elapsed time: ... ms

    """

    def __init__(self, log=None, msg=None):
        # Define a custom msg if needed
        self.msg = "Entering timed block..." if msg is None else msg
        # We store a shallow copy of the input logger
        self.logger = log

    @property
    def step(self):
        """Return time elapsed from timer start (milliseconds)."""
        return (time.time() - self.start) * 1000  # Interval as milliseconds

    def __enter__(self):
        """Called upon before entering a 'with' block."""
        self.start = time.time()
        # Logging timer start if needed
        if self.logger is not None:
            self.logger.debug(self.msg)
        else:
            print(self.msg)
        # This allow using of 'as' statement (e.g.: with self.timer() as t)
        return self

    def __exit__(self, type, value, traceback):
        """Called upon before exit from 'with' block."""
        self.end = time.time()
        self.interval = (self.end - self.start) * 1000  # Interval as ms
        # Logging timer end if needed
        if self.logger is not None:
            self.logger.debug("Elapsed time: " + str(self.interval) + " ms")
        else:
            print("Elapsed time: {:} ms".format(self.interval))

    @staticmethod
    def timed(log=None, msg=None):
        """Timer decorator.

        Returns a decorator that can be used to measure
        execution time of any method.
        As this decorator accepts optional arguments,
        must be called as a method. See examples.

        Parameters
        ----------
        log : CLog or None, optional
            Instance of :class:`.CLog` class to be used as
            performance logger. If a logger is specified,
            timer data will not be printed on console.
        msg : str or None, optional
            Custom message to display when entering the timed block.
            If None, "Entering timed block..." will printed.

        See Also
        --------
        .CLog : CLog and store runtime information on disk.

        Examples
        --------
        >>> from secml.array import CArray
        >>> from secml.utils import CTimer

        >>> @CTimer.timed()
        ... def abc():
        ...     print("Hello world!")

        >>> abc()  # doctest: +ELLIPSIS
        Entering timed block `abc`...
        Hello world!
        Elapsed time: ... ms

        """
        def wrapper(fun):
            @wraps(fun)  # To make wrapped_fun work as fun
            def wrapped_fun(*args, **kwargs):
                # Setting a custom message
                msg_a = msg
                if msg is None:
                    msg_a = "Entering timed block " \
                            "`{:}`...".format(fun.__name__)
                # Execute the function with the timer
                with CTimer(log=log, msg=msg_a):
                    return fun(*args, **kwargs)
            return wrapped_fun
        return wrapper
