import wandb
from lob_simulation import multiple_runs

# Your target stylised facts
TARGET_KURT = 3
TARGET_ACF = [0.20, 0.20, 0.20, 0.20]


def objective(metrics):
    err = 0

    err += (metrics["avg_kurt"] - TARGET_KURT) ** 2

    for i in range(4):
        err += (metrics[f"acf_lag{i+1}"] - TARGET_ACF[i]) ** 2

    return err


def train():
    run = wandb.init()

    config = dict(wandb.config)

    parameters = {
        "save_path": config["save_path"],

        "N": config["N"],
        "w": config["w"],
        "Pf": config["Pf"],
        "S": config["S"],
        "deltaP": config["deltaP"],
        "tc": config["tc"],
        "TE": config["TE"],
        "Pmin": config["deltaP"],
        "Pmax": 2*(config["Pf"] + config["S"]),
        "L": config["L"],
        "D": config["D"],
        "R": config["R"],
        "BURN": int(round(0.1*config["TE"])),

        "ALPHA": config["ALPHA"],
        "GAMMA": config["GAMMA"],
        "EPS0": config["EPS0"],
        "EPS_DECAY": config["EPS_DECAY"],
        "EPS_MIN": config["EPS_MIN"],

        "delta_l": config["deltaP"],

        "LAMBDA_INV": config["LAMBDA_INV"],
        "LAMBDA_HOLD": config["LAMBDA_HOLD"],
    }

    metrics = multiple_runs(parameters)

    err = objective(metrics)

    wandb.log({
        "err": err,
        **metrics
    })

