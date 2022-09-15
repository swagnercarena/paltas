import argparse
import inspect

import docstring_parser


def make_cli(main_f):
    """Wrap main_f in an argparse command-line interface

    Docstring and type annotations determine argument parsing and help message.
    """
    # Get the signature, used for getting argument names and types
    signature = inspect.signature(main_f).parameters

    # Parse the docstring to get the argument descriptions
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
        kwargs = dict(help=descs.get(pname, None))
        has_default = param.default != param.empty
        if has_default and param.annotation is bool:
            # Flag argument. Note true/false inversion, gotta love argparse
            if param.default is True:
                kwargs['action'] = 'store_false'
            elif param.default is False:
                kwargs['action'] = 'store_true'
            else:
                raise ValueError("flag args should default to true or false")
        else:
            # Regular argument
            if has_default:
                kwargs['default'] = param.default
            if param.annotation != param.empty:
                kwargs['type'] = param.annotation
            if param.kind == param.VAR_POSITIONAL:
                kwargs['nargs'] = '*'
        parser.add_argument(('--' if has_default else '') + pname, **kwargs)

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
