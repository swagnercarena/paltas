#!/usr/bin/env python
import argparse
from pathlib import Path
import random
import shutil
import string

import numpy as np
import paltas
import paltas.Analysis

config_header = """\
from paltas.Configs.paper_2203_00690.config_test_10 import *
config_dict = copy.deepcopy(config_dict)
"""


def robustness_test(
        param_code: str, 
        param_value, 
        n_images=5, 
        model_path='./xresnet34_full_marg_1_final.h5',
        norm_path='./norms.csv',
        n_mcmc_samples=int(1e3)):
    """Run an end-to-end robustness test.

    Arguments:
     - param_code: name of parameter to change. Use / for nesting,
        e.g. subhalo/parameters/sigma_sub
     - param_value: value you wish the parameter to take
     - n_images: images to generate
     - model_path: path to neural network h5 file
     - norm_path: path to norms.css file
     - n_mcmc_samples: number of MCMC samples to do 
        (excluding 1k burn-in samples)

    This script will:    
     - Creates a configuration that differes by one parameter value
        from a reference config (paper_2203_00690.config_test_10).
     - Generates and saves test set images with that config
     - Runs a neural network over that config, saving results
     - Runs Bayesian MCMC and asymptotic frequentist hierarchical inference,
        saving results.

    Results are saved in the current directory, in the following folders:
     - Config: robustness_test_configs/DATASET_NAME.py
     - Images: robustness_test_images/DATASET_NAME
     - Network outputs: robustness_test_results/DATASET_NAME_network_outputs.npz
     - Inference: robustness_test_results/DATASET_NAME_inference_results.npz
    """
    
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
        
    param_name = param_code.replace('/', '_')
    # Can't have dots in the dataset name, not legal python module name
    dataset_name = f"{param_name}:{param_value}"

    # Create config
    config = (
        config_header 
        + "\nconfig_dict['" 
        + param_code.replace('/', "']['") + f"'] = {param_value}")
    # Write it to a .py file
    # Note we can't use the dataset name as the config name;
    # the param value may contain nasty characters like .
    # (which python will see as a module separator)
    config_name = ''.join(random.choices(string.ascii_lowercase, k=16))
    config_fn = config_folder / f"{config_name}.py"
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
    parser = argparse.ArgumentParser(usage=robustness_test.__doc__)
    # Repeating arguments like this is ugly. Libraries like fire
    # will do this automatically, but would add a dependency to paltas.
    parser.add_argument("param_code", type=str)
    parser.add_argument("param_value", type=str)
    parser.add_argument("--n_images", type=int, default=5)
    parser.add_argument(
        "--model_path",
        type=str, 
        default='./xresnet34_full_marg_1_final.h5')
    parser.add_argument(
        "--norm_path",
        default='./norms.csv')
    parser.add_argument(
        "--n_mcmc_samples",
        default=int(1e3),
        type=int)
    args = parser.parse_args()

    robustness_test(
        args.param_code, 
        args.param_value,
        n_images=args.n_images,
        model_path=args.model_path,
        norm_path=args.norm_path,
        n_mcmc_samples=args.n_mcmc_samples)
