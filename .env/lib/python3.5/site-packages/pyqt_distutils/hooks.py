"""
This module contains the hooks load and our builtin hooks.

"""
import re
import pkg_resources

from .utils import write_message


#: Name of the entrypoint to use in setup.py
ENTRYPOINT = 'pyqt_distutils_hooks'


def load_hooks():
    """
    Load the exposed hooks.

    Returns a dict of hooks where the keys are the name of the hook and the
    values are the actual hook functions.
    """
    hooks = {}
    for entrypoint in pkg_resources.iter_entry_points(ENTRYPOINT):
        name = str(entrypoint).split('=')[0].strip()
        try:
            hook = entrypoint.load()
        except Exception as e:
            write_message('failed to load entry-point %r (error="%s")' % (name, e), 'yellow')
        else:
            hooks[name] = hook
    return hooks


def hook(ui_file_path):
    """
    This is the prototype of a hook function.
    """
    pass


def gettext(ui_file_path):
    """
    Let you use gettext instead of the Qt tools for l18n
    """
    with open(ui_file_path, 'r') as fin:
        content = fin.read()

    # replace ``_translate("context", `` by ``_(``
    content = re.sub(r'_translate\(".*",\s', '_(', content)
    content = content.replace(
        '        _translate = QtCore.QCoreApplication.translate', '')

    with open(ui_file_path, 'w') as fout:
        fout.write(content)
