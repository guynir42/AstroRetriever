"""
Various utility functions and classes
that were not relevant to any specific module.
"""


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
