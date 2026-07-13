"""
"""

import ast
import re

# PERF PATCH (26-07-14): a bare unit name (only [A-Za-z_][A-Za-z0-9_]*) parses to a single ast.Name
# and eval()s to the ONE pre-registered singleton object — identity-stable across calls. Only such
# labels are memoized below, so the cache returns the exact object the original would. Compound
# expressions ('m/s', 'g/cc') eval to a FRESH object each call; caching them would change identity,
# so they deliberately fall through to the original path.
_simple_name_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_MISS = object()  # sentinel: distinguishes a real cached value from an absent key


class UnitRegistry:
    # Note that this structure ensures that UnitRegistry behaves as a singleton

    class __Registry:

        __shared_state = {}
        whitelist = (
            ast.Expression,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.BinOp,
            ast.UnaryOp,
            ast.operator,
            ast.unaryop,
        )

        def __init__(self):
            self.__dict__ = self.__shared_state
            self.__context = {}

        def __getitem__(self, string):
            # This approach to avoiding arbitrary evaluation of code is based on https://stackoverflow.com/a/11952618 
            # by https://stackoverflow.com/users/567292/ecatmur
            stripped_string = string.strip()  # discard leading or trailing spaces before parsing
            tree = ast.parse(stripped_string, mode="eval")
            valid = all(isinstance(node, self.whitelist) for node in ast.walk(tree))
            if valid:
                try:
                    item = eval(
                        compile(tree, filename="", mode="eval"),
                        {"__builtins__": {}},
                        self.__context,
                    )
                except NameError:
                    raise LookupError('Unable to parse units: "%s"' % string)
                else:
                    return item
            else:
                # could return self['UnitQuantity'](string)
                raise LookupError('Unable to parse units: "%s"' % string)

        def __setitem__(self, string, val):
            assert isinstance(string, str)
            try:
                assert string not in self.__context
            except AssertionError:
                if val == self.__context[string]:
                    return
                raise KeyError(
                    '%s has already been registered for %s'
                    % (string, self.__context[string])
                )
            self.__context[string] = val

    __regex = re.compile(r'([A-Za-z])\.([A-Za-z])')
    __registry = __Registry()

    def __getattr__(self, attr):
        return getattr(self.__registry, attr)

    def __setitem__(self, label, value):
        self.__registry.__setitem__(label, value)

    def __getitem__(self, label):
        """Parses a string description of a unit e.g., 'g/cc'"""

        # PERF PATCH (26-07-14): memoize bare-name lookups by raw label. A bare name resolves to the
        # registry's pre-registered singleton (identity-stable — see _simple_name_re note above), so
        # caching is behavior-identical to the original. This removes a full regex-normalize +
        # ast.parse + compile + eval on the hot path: in the neo14 P1 get_data cross-rank merge,
        # Dimensionality.__hash__ does unit_registry['dimensionless'] on every hash (~4.8M calls /
        # ~168 s of pure redundant re-parsing). Compound expressions are NOT cached (they eval fresh
        # each call — caching would change object identity); a LookupError also propagates uncached,
        # so a unit registered later still resolves.
        cacheable = _simple_name_re.match(label) is not None
        if cacheable:
            try:
                cache = self.__memo
            except AttributeError:
                cache = self.__memo = {}
            hit = cache.get(label, _MISS)
            if hit is not _MISS:
                return hit

        norm = self.__regex.sub(
            r"\g<1>*\g<2>", label.replace('^', '**').replace('·', '*'))

        # make sure we can parse the label ....
        if norm == '': norm = 'dimensionless'
        if "%" in norm: norm = norm.replace("%", "percent")
        if norm.lower() == "in": norm = "inch"

        result = self.__registry[norm]
        if cacheable:
            cache[label] = result
        return result

unit_registry = UnitRegistry()
