try:
    import colorama
except ImportError:
    has_colorama = False
else:
    has_colorama = True

import shlex
try:
    # Python 3
    from shlex import quote
except ImportError:
    # Python 2
    from pipes import quote


def build_args(cmd, src, dst):
    """
        Build arguments list for passing to subprocess.call_check

        :param cmd str: Command string to interpolate src and dst filepaths into.
            Typically the output of `config.Config.uic_command` or `config.Config.rcc_command`.
        :param src str: Source filepath.
        :param dst str: Destination filepath.
    """
    cmd = cmd % (quote(src), quote(dst))
    args = shlex.split(cmd)

    return [arg for arg in args if arg]


def write_message(text, color=None):
    if has_colorama:
        colors = {
            'red': colorama.Fore.RED,
            'green': colorama.Fore.GREEN,
            'yellow': colorama.Fore.YELLOW,
            'blue': colorama.Fore.BLUE
        }
        try:
            print(colors[color] + text + colorama.Fore.RESET)
        except KeyError:
            print(text)
    else:
        print(text)
