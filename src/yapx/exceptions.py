x = 5


class ArgumentConflictError(Exception):
    ...


class ParserClosedError(Exception):
    ...


class UnsupportedTypeError(Exception):
    ...


class NoArgsModelError(Exception):
    ...


class MutuallyExclusiveArgumentError(Exception):
    ...


class MutuallyExclusiveRequiredError(Exception):
    ...
