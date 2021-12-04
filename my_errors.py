#!/usr/bin/env python

# -----------------------------------------------------------
# Involved customized errors of gigaword & HC summary.
# This file serves as the helper file for the project "unsupervised sentence summarization"
# which is supervised by Dr. Mou (Lili Mou).
# (C) 2021 Puyuan Liu, Department of Computing Science, University of Alberta
# Released under GNU Public License (GPL)
# email puyuan@ualberta.ca
# -----------------------------------------------------------

class MyError(Exception):
    """Base Exception class to raise errors for the project.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class SelfConflictingError(MyError):
    def __init__(self, message):
        super().__init__(message)

    def print_error(self):
        print(self.message)


