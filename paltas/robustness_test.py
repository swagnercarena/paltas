#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import re
import shutil
import string
import inspect

import docstring_parser


config_header = """\
import sys
sys.path.append('{CONFIG_PATH}')
from {CONFIG_NAME} import *

config_dict = copy.deepcopy(config_dict)
"""


def robustness_test(
        *param_value_pairs,
        n_images: int = 5,
        n_rotations: int = 32,
        config_path: str = None,
        model_path: str = './xresnet34_full_marg_1_final.h5',
        norm_path: str = './norms.csv',
        n_mcmc_samples: int = 1000):
    """Run an end-to-end test of paltas image generation and analysis.

    This script will:    
     - Create a configuration that differes by one parameter value
       from the reference config;
     - Generate and saves test set images with that config;
     - Run a neural network over that config, saving results;
     - Run Bayesian MCMC and asymptotic frequentist hierarchical inference,
       saving and printing results.

    Results are saved in the current directory, in the following folders:
     - Config: robustness_test_configs/DATASET_NAME.py
     - Images: robustness_test_images/DATASET_NAME
     - Network outputs: robustness_test_results/DATASET_NAME_network_outputs.npz
     - Inference: robustness_test_results/DATASET_NAME_inference_results.npz

    Args:
        param_value_pairs: Flat list with parameter/value pairs to set.
            Omit to just run one config. When naming parameters, use / to 
            indicate taking a key, e.g. subhalo/parameters/sigma_sub.
        param_value: value you wish the parameter to take
        n_images: number of images to generate
        n_rotations: average network predictions over n_rotations image
            rotations.
        config_path: path to paltas config py file for base settings
            Defaults to paper_2203_00690.config_val
        model_path: path to neural network h5 file
        norm_path: path to norms.css file
        n_mcmc_samples: number of MCMC samples to do 
            (excluding 1k burn-in samples)
    """
    # Delayed imports, to make sure --help calls finish quickly
    import numpy as np
    import paltas
    import paltas.Analysis

    # Use default validation set config if no config given
    if config_path is None:
        config_path = (
            paltas.Analysis.gaussian_inference.DEFAULT_TRAINING_SET.parent
            / 'config_val.py')
    config_path = Path(config_path)

    # Check we have the network and norms.csv.
    # (and if not, crash now before expensive image generation)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model h5 file {model_path} not found")
    if not Path(norm_path).exists():
        raise FileNotFoundError(f"Norms csv file {norm_path} not found")
    
    # Make sure base folders exist
    config_folder = Path('./robustness_test_configs')
    images_base_folder = Path('./robustness_test_images')
    results_folder = Path('./robustness_test_results')
    for x in (config_folder, images_base_folder, results_folder):
        x.mkdir(exist_ok=True)

    # Group param_value pairs    
    if len(param_value_pairs) % 2 != 0:
        raise ValueError("param_value_pairs must have an even number of items")
    pv_pairs = [
        param_value_pairs[pos:pos + 2]
        for pos in range(0, len(param_value_pairs), 2)]
    
    # Construct dataset_name
    dataset_name = config_path.stem
    # Remove config, and make sure we start with a character
    # (so the dataset is a valid python module name)
    if dataset_name.startswith('config_'):
        dataset_name = dataset_name[len('config_'):]
    if dataset_name[0].isdigit():
        dataset_name = '_' + dataset_name
    # Add safe versions of the key/value pairs
    for param_code, param_value in pv_pairs:
        dataset_name += '_' + '_'.join([
            re.sub('[^a-zA-Z0-9]', '_', x)
            for x in (param_code, str(param_value))
        ])
    # Ensure dataset is <100 characters, and add a random suffix for uniqueness
    dataset_name = (
        dataset_name[:92]
        + '_' 
        + ''.join(random.choices(string.ascii_lowercase, k=8))
    )
    

    # Create config and write it to a .py file
    config = config_header.format(
        CONFIG_PATH=str(config_path.parent),
        CONFIG_NAME=config_path.name.split('.')[0])
    for param_code, param_value in pv_pairs:
        config += (
            "\nconfig_dict['" + 
            param_code.replace('/', "']['") + f"'] = {param_value}")
    config_fn = config_folder / f"{dataset_name}.py"
    with open(config_fn, mode='w') as f:
        f.write(config)

    # Generate images
    print(f"Generating images from {config_fn}\n\n")
    dataset_folder = images_base_folder / dataset_name
    if dataset_folder.exists():
        print(f"Removing existing dataset folder {dataset_folder}")
        shutil.rmtree(dataset_folder)
    paltas.generate.generate_from_config(
        config_path=str(config_fn),
        save_folder=str(dataset_folder),
        n=n_images,
        save_png_too=False,
        # Will be generated later
        tf_record=False)

    # Run neural network, copy out results
    print(f"\n\nRunning neural network on {dataset_folder}\n\n")
    paltas.Analysis.gaussian_inference.run_network_on(
        dataset_folder,
        norm_path=norm_path,
        model_path=model_path,
        batch_size=min(n_images, 50),
        n_rotations=n_rotations,
        save_penultimate=False)
    shutil.copy(
        src=dataset_folder / 'network_outputs.npz',
        dst=results_folder / f"{dataset_name}_network_outputs.npz")

    # Run inference
    print(f"\n\nRunning final inference\n\n")
    inf = paltas.Analysis.gaussian_inference.GaussianInference.from_folder(dataset_folder)
    freq_summary, freq_cov = inf.frequentist_asymptotic()
    bayes_summary, chain = inf.bayesian_mcmc(n_samples=n_mcmc_samples)

    # Combine results into one dataframe
    summary = freq_summary[['param', 'truth']].copy()
    for df, code in ((freq_summary, 'maxlh'), (bayes_summary, 'mcmc')):
        summary[f'{code}_fit'] = df['fit']
        summary[f'{code}_fit_unc'] = df['fit_unc']

    # Save and print inference results
    np.savez(
        results_folder / f"{dataset_name}_inference_results.npz",
        summary=summary.to_records(),
        freq_cov=freq_cov,
        chain=chain)
    print(f"\n\nDone!! :-)\n\n")
    print(f"RESULTS:\n")
    try:
        print(summary.to_markdown(index=False))
    except ImportError:
        print(summary)
        print("\n\nFor prettier result prints, pip install tabulate.")


if __name__ == '__main__':
    main_f = robustness_test

    # Get the signature, will get argument names and types later
    signature = inspect.signature(main_f).parameters
    pname_list = list(signature.keys())
    first_pname, other_pnames = pname_list[0], pname_list[1:]

    # Parse the docstring to get the argument types
    doc = docstring_parser.parse(main_f.__doc__)
    descs = {x.arg_name: x.description for x in doc.params}

    # Auto-generate and runthe argparse parser
    # There are libraries that do this, but I haven't found one that parses
    # the docstring for argument descriptions.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=doc.short_description + "\n\n" + doc.long_description)
    for pname, param in signature.items():
        has_default = (param.default != param.empty)
        parser.add_argument(
            ('--' if has_default else '') + pname,
            nargs='*' if param.kind == param.VAR_POSITIONAL else 1,
            help=descs[pname],
            type=param.annotation if has_default else None,
            default=param.default if has_default else tuple())

    args = parser.parse_args()

    # If there is an *args we need to call the main function differently
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
