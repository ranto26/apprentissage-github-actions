from pathlib import Path

score = 0.92  # Valeur fictive pour le TP

Path("metrics.txt").write_text(
    f"accuracy={score}\n",
    encoding="utf-8",
)

print(f"Accuracy : {score}")
