"""
Various utility functions and classes
that were not relevant to any specific module.
"""
import sys


class OnClose:
    """
    Create an instance of this class so that it
    runs the given function/lambda when it goes
    out of scope.
    This could be useful for removing files,
    deleting things from the DB, and so on.
    It triggers even if there is an exception,
    so it is kind of like a finally block.
    """

    def __init__(self, func):
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func

    def __del__(self):
        self.func()


def trim_docstring(docstring):
    """
    Remove leading and trailing lines, remove indentation, etc.
    See PEP 257: https://peps.python.org/pep-0257/
    """
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def short_docstring(docstring):
    """
    Get the first line of the docstring.
    Assumes the docstring has already been cleared
    of leading new lines and indentation.
    """
    if not docstring:
        return ""

    return docstring.splitlines()[0]


def help_with_class(cls, pars_cls=None, sub_classes=None):
    """
    Print the help for this object and objects contained in it.

    Parameters
    ----------
    cls : class
        The class to print help for.
    pars_cls : class, optional
        The class that contains the parameters for this class.
        If None, no parameters are printed.
    sub_classes : list of classes, optional
        A list of classes that are contained in this class.
        The help for each of those will be printed.
    """
    description = short_docstring(trim_docstring(cls.__doc__))

    print(f"{cls.__name__}\n" "--------\n" f"{description}")
    if pars_cls is not None:
        print("Parameters:")
        # initialize a parameters object and print it
        pars = pars_cls(cfg_file=False)  # do not read config file
        pars.print()  # show a list of parameters

        print()  # newline

    if sub_classes is not None:
        for sub_cls in sub_classes:
            if hasattr(sub_cls, "help") and callable(sub_cls.help):
                sub_cls.help()

        print()  # newline
