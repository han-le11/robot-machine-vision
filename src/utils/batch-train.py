import subprocess
import itertools

#  Hyperparameter grid 
seq_lens = [16]
batch_sizes = [1, 2, 4, 6, 8]

#  Loop through all combinations 
for seq_len, batch_size in itertools.product(seq_lens, batch_sizes):
    print("\n==============================")
    print(f"Training with SEQ_LEN={seq_len}, BATCH_SIZE={batch_size}")
    print("==============================")

    cmd = [
        "python", "-m", "src.train.train",
        f"--seq_len={seq_len}",
        f"--batch_size={batch_size}"
    ]

    subprocess.run(cmd)
