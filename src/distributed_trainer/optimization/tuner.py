import optuna
from loguru import logger

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64])
    logger.info(f"Evaluating Trial {trial.number} with lr={lr}, hidden_dim={hidden_dim}")
    return 1.0

def run_hpo_study():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    logger.info(f"Best trial: {study.best_trial.value} with params {study.best_params}")
    return study
