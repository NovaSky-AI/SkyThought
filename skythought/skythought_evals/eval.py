import argparse
import json
import os
import subprocess

from .tasks import TASK_NAMES_TO_YAML


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process model path, prompt format, and evals to run."
    )
    parser.add_argument("--model", required=True, type=str, help="Path to the model.")
    parser.add_argument(
        "--evals",
        required=True,
        type=str,
        help=f"Comma-separated list of evals to run (no spaces). We currently support the following tasks {list(TASK_NAMES_TO_YAML.keys())}",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Difficulty for the dataset. Options: 'easy', 'medium', 'hard'",
    )
    parser.add_argument("--subset", type=str, help="Subset for the dataset.")
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Output file to write results to.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0],
        help="Temperature for sampling.",
    )
    return parser.parse_args()


def extract_accuracy_from_output(output):
    # Iterate through all lines from the end to the beginning
    lines = output.splitlines()[::-1]
    for line in lines:
        try:
            # Attempt to parse a JSON object from the line
            data = json.loads(line.replace("'", '"'))
            if "acc" in data:
                return data["acc"]
        except json.JSONDecodeError:
            continue
    return None


def write_logs_to_file(logs, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(logs)
        print(f"Logs successfully written to {output_file}")
    except IOError as e:
        print(f"Failed to write logs to file {output_file}: {e}")


def main():
    args = parse_arguments()

    # Extract the arguments
    model_path = args.model
    evals = args.evals.split(",")
    output_file = args.output_file
    tp = args.tp
    temperatures = [str(t) for t in args.temperatures]

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_and_check.py"
    )

    # Hold all logs
    all_logs = ""
    results = {}

    # Run the Python command for each eval and collect logs
    for eval_name in evals:
        eval_name = eval_name.lower()
        assert (
            eval_name in TASK_NAMES_TO_YAML.keys()
        ), f"Task {eval_name} not found, should be one of {TASK_NAMES_TO_YAML.keys()}"
        command = [
            "python",
            script_path,
            "--model",
            model_path,
            "--task",
            eval_name,
            "--tp",
            str(tp),
            "--temperatures",
        ]
        command.extend(temperatures)  # Add temperatures as separate arguments

        if args.difficulty:
            command.append("--difficulty")
            command.append(args.difficulty)

        print(f"Running eval {eval_name} with command {command}")
        all_logs += f"\nRunning eval: {eval_name} with command {command}\n"
        try:
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as proc:
                output_lines = []
                for line in proc.stdout:
                    print(line, end="")  # Stream output to the console
                    output_lines.append(line)
                    all_logs += line
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, command)

                # Capture output for post-processing
                output = "".join(output_lines)
                accuracy = extract_accuracy_from_output(output)
                results[eval_name] = accuracy

        except subprocess.CalledProcessError as e:
            error_message = f"Error occurred while running eval {eval_name}: {e}\n"
            print(error_message)
            all_logs += error_message

    # Write logs of all stdout / stderr to a file
    write_logs_to_file(all_logs, output_file)

    print("Results:")
    print(results)


if __name__ == "__main__":
    main()
