import importlib
import pkgutil
import sys
try:
    import importlib as imp
except ImportError:
    import imp

# Discover the plugins (must be named 'secml_{plugin}')
# And also import the respective module a first time
secml_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules()
    if name.startswith('secml_')
}

# Attach each plugin to the `secml.ext` namespace
for plugin in secml_plugins:
    plugin_name = plugin.lstrip('secml_')
    plugin_path = secml_plugins[plugin].__path__
    if len(plugin_path) != 1:
        raise RuntimeError("plugins defined in multiple directories are "
                           "not supported (check '{:}')".format(plugin))
    plugin_init = secml_plugins[plugin].__file__
    module_name = __name__ + '.' + plugin_name
    if module_name in sys.modules:
        raise RuntimeError("plugin '{:}' already defined?!".format(plugin))
    if sys.version_info < (3, 4):  # imp is deprecated since Py3.4
        imp.load_package(module_name, plugin_path[0])
    else:
        spec = imp.util.spec_from_file_location(
            module_name, plugin_init,
            submodule_search_locations=[])
        module = imp.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
