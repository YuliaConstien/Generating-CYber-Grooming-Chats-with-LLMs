import subprocess

def run_training():
    subprocess.run(["python", "train_model.py"])

def run_evaluation(method):
    subprocess.run(["python", "evaluate_model.py", method])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training or evaluation for the model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model.")
    parser.add_argument("--method", type=str, choices=["--all", "--f1", "--classification_report", "--confusion_matrix", "--roc"], default="--all", help="Evaluation method.")

    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.evaluate:
        run_evaluation(args.method)
    else:
        print("Please specify --train or --evaluate")
