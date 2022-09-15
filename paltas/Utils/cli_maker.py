import argparse
import inspect

import docstring_parser


def make_cli(main_f):
    """Wrap main_f in an argparse command-line interface

    Docstring and type annotations determine argument parsing and help message.
    """
    # Get the signature, will get argument names and types later
    signature = inspect.signature(main_f).parameters
    pname_list = list(signature.keys())

    # Parse the docstring to get the argument types
    doc = docstring_parser.parse(main_f.__doc__)
    descs = {x.arg_name: x.description for x in doc.params}

    # Auto-generate and and run an argparse parser
    # There are libraries that do this, but I haven't found one that parses
    # the docstring for argument descriptions.
    desc = doc.short_description
    if doc.long_description:
        desc += "\n\n" + doc.long_description
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc)
    for pname, param in signature.items():
        has_default = (param.default != param.empty)
        parser.add_argument(
            ('--' if has_default else '') + pname,
            nargs='*' if param.kind == param.VAR_POSITIONAL else None,
            help=descs[pname],
            type=param.annotation if has_default else None,
            default=param.default if has_default else tuple())

    args = parser.parse_args()

    # If there is a *args, we need to call the main function differently
    var_pos_name, = [
            pname for pname, param in signature.items() 
            if param.kind == param.VAR_POSITIONAL
        ][:1] or (None,)
    if var_pos_name:
        var_pos = getattr(args, var_pos_name)
        if not var_pos:
            var_pos = tuple()
        main_f(
            *var_pos,
            **{pname: getattr(args, pname) 
               for pname in signature.keys()
               if pname != var_pos_name})
    else:
        main_f(**vars(args))
