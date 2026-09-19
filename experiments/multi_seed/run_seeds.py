import os
import subprocess
import torch


SEEDS = [
    42,
    52,
    62
]


results_file = "experiments/multi_seed/results.txt"


os.makedirs(
    "experiments/multi_seed",
    exist_ok=True
)



with open(results_file, "w") as f:

    f.write(
        "Multi Seed QVAE Validity Evaluation\n\n"
    )



for seed in SEEDS:

    print("\n==============================")
    print(f"Running Seed: {seed}")
    print("==============================\n")


    # Run training with seed override
    subprocess.run(
        [
            "python",
            "experiments/qvae_cross_attention.py",
            "--seed",
            str(seed)
        ]
    )


    # Run validity evaluation

    output = subprocess.check_output(
        [
            "python",
            "experiments/validity_evaluation.py"
        ],
        text=True
    )


    print(output)



    with open(results_file,"a") as f:

        f.write(
            f"\n\nSeed {seed}\n"
        )

        f.write(output)



print(
    "\nMulti seed experiments completed."
)