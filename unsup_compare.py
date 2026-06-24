import argparse
import logging
import os
import matplotlib.pyplot as plt

# Import comparison models
from modules.comparisons import (
    run_source_only, run_standard_pseudo_labeling, run_cotta, run_t_uda,
    run_lidar_uda, run_gipso, run_train_till_you_drop, run_unsup_gated_adapters,
    run_xmuda, run_annotator, run_bi3d, run_tent
)

# Import HDC baseline logic
import unsup_experiment as hdc_exp

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def save_graphic(save_path, title, results_dict):
    """
    Saves a graphic plotting the results of multiple models.
    results_dict: dict mapping model_name -> data_list OR data_dict
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    for model_name, data in results_dict.items():
        if isinstance(data, dict):
            for metric_name, values in data.items():
                plt.plot(values, label=f"{model_name} ({metric_name})")
        elif data:
            plt.plot(data, label=model_name)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Metric')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Model Comparisons for Unsupervised & Active Domain Adaptation Experiments")
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the models if applicable')
    parser.add_argument('--pretrained_path', type=str, default='checkpoints/', help='Base path for pretrained models')
    parser.add_argument('--log_dir', type=str, default='logs/comparisons', help='Directory to save in-depth logs and graphics')
    
    # Models flag
    parser.add_argument('--models', nargs='+', default=['hdc'], 
                        help='List of models to compare. E.g., --models hdc cotta source_only t_uda')

    # Experiment flags (same as unsup_experiment.py)
    parser.add_argument('--intra_weather', action='store_true', help='Run Intra-Dataset UDA (Weather)')
    parser.add_argument('--inter_geography', action='store_true', help='Run Inter-Dataset UDA (Geography)')
    parser.add_argument('--inter_density', action='store_true', help='Run Inter-Dataset UDA (Density)')
    parser.add_argument('--ada', action='store_true', help='Run Active Domain Adaptation (ADA)')
    parser.add_argument('--online_stability', action='store_true', help='Run Online Stability over Time (Ablation)')
    parser.add_argument('--compute_efficiency', action='store_true', help='Run Compute & Efficiency Cost (Ablation)')
    
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, 'comparison_run.log')
    logger = setup_logger(log_file)
    logger.info(f"Started Comparisons for models: {args.models}")

    # Map model names to their run functions
    model_runners = {
        'source_only': run_source_only,
        'standard_pseudo_labeling': run_standard_pseudo_labeling,
        'cotta': run_cotta,
        't_uda': run_t_uda,
        'lidar_uda': run_lidar_uda,
        'gipso': run_gipso,
        'train_till_you_drop': run_train_till_you_drop,
        'unsup_gated_adapters': run_unsup_gated_adapters,
        'xmuda': run_xmuda,
        'annotator': run_annotator,
        'bi3d': run_bi3d,
        'tent': run_tent,
        # 'hdc' is handled directly via unsup_experiment
    }

    # Gather experiments to execute
    experiments = []
    if args.intra_weather: experiments.append(('intra_weather', 'Intra-Dataset UDA (Weather)'))
    if args.inter_geography: experiments.append(('inter_geography', 'Inter-Dataset UDA (Geography)'))
    if args.inter_density: experiments.append(('inter_density', 'Inter-Dataset UDA (Density)'))
    if args.ada: experiments.append(('ada', 'Active Domain Adaptation (ADA)'))
    if args.online_stability: experiments.append(('online_stability', 'Online Stability over Time'))
    if args.compute_efficiency: experiments.append(('compute_efficiency', 'Compute & Efficiency Cost'))

    for exp_key, exp_title in experiments:
        logger.info(f"Running {exp_title} for models: {args.models}")
        results = {}
        for m in args.models:
            if m == 'hdc':
                # Load HDC Baseline
                if args.pretrain:
                    model_instance = hdc_exp.pretrain_hdc_model()
                else:
                    hdc_path = os.path.join(args.pretrained_path, 'hdc_baseline.pt')
                    model_instance = hdc_exp.load_hdc_model(hdc_path)
                
                # Execute specific HDC experiment
                if exp_key == 'intra_weather': results[m] = hdc_exp.run_intra_weather(model_instance, logger)
                elif exp_key == 'inter_geography': results[m] = hdc_exp.run_inter_geography(model_instance, logger)
                elif exp_key == 'inter_density': results[m] = hdc_exp.run_inter_density(model_instance, logger)
                elif exp_key == 'ada': results[m] = hdc_exp.run_ada(model_instance, logger)
                elif exp_key == 'online_stability': results[m] = hdc_exp.run_online_stability(model_instance, logger)
                elif exp_key == 'compute_efficiency': results[m] = hdc_exp.run_compute_efficiency(model_instance, logger)
                
            elif m in model_runners:
                logger.info(f"Running {m} for {exp_title}...")
                results[m] = model_runners[m](data=None, experiment=exp_key, logger=logger)
            else:
                logger.warning(f"Model '{m}' not recognized. Skipping.")

        # Save comparative graphic for this experiment
        save_graphic(os.path.join(args.log_dir, f'{exp_key}_comparison.png'), exp_title, results)

if __name__ == "__main__":
    main()
