from datetime import datetime
import numpy as np
from src.data.generate_data import main as generate_data
from src.xgboost.training.train import main as train
from src.xgboost.thresholds.compute_thresholds import main as compute_thresholds
from src.data.generate_test_data import generate_test_data
from src.xgboost.inference.infer import run_inference
from src.xgboost.evaluation.evaluate import evaluate
from src.common.load_main_config import load_data_config


def main():

    print("\n===== STEP 1: GENERATE TRAIN DATA =====")
    generate_data()

    print("\n===== STEP 2: TRAIN MODEL =====")
    train()

    print("\n===== STEP 3: COMPUTE THRESHOLDS =====")
    compute_thresholds()

    print("\n===== STEP 4: GENERATE TEST DATA =====")
    config = load_data_config()
    np.random.seed(None)
    df_test = generate_test_data(
        start_date=datetime(2025, 4, 1),
        hours=48,
        config=config
    )

    print("\n===== STEP 5: RUN INFERENCE =====")
    results = run_inference(df_test)

    print("\n===== STEP 6: EVALUATION =====")
    evaluate(results)


if __name__ == "__main__":
    main()