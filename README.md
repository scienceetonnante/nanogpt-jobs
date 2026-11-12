# Train a small GPT on a Hugging Face GPU, without changing your code

`train.py` trains a GPT-style language model with 51M parameters on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a collection of 2 million short stories for young children. The script runs as it is on a Mac, on an NVIDIA GPU or on a CPU. On a MacBook Pro (M4 Pro), training takes about 12 hours and keeps the GPU busy the whole time.

`run_on_hf.sh` runs the same file, unchanged, on an A100 GPU on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). Your computer stays free, and the run takes about 30 minutes.

## Files

| File | Role |
|---|---|
| `requirements.txt` | Lists the Python packages the code needs |
| `train.py` | Downloads the data, defines the model, trains it, saves it |
| `run_on_hf.sh` | Runs a Python file of this folder on a Hugging Face GPU |
| `sample.py` | Writes stories with the trained model |

## Run on your machine

```bash
uv run --with-requirements requirements.txt train.py
```

[uv](https://docs.astral.sh/uv/) installs the packages listed in `requirements.txt`, then runs `train.py`. The model is saved to `out/model.pt`. Every 100 steps the script prints the loss, the speed and the time left. Every 2,000 steps it prints a story written by the model.

## Run on a Hugging Face GPU

You need a Hugging Face account with a PRO subscription or credits, and you need to be logged in (`hf auth login`).

```bash
./run_on_hf.sh train.py                # on an A100, stopped after 2 hours at most
./run_on_hf.sh train.py l4x1 4h        # on another GPU, with another time limit
```

The script does three things:

1. It copies this folder to a private storage bucket on the Hub, `<you>/nanogpt-jobs`, in the folder `code/`. Hidden files and folders (`.git`, `.venv`, …), `__pycache__` and `out/` are left out.
2. It starts a GPU machine, copies the code there, and turns `out/` into a link to the folder `out/` of the bucket. Everything the code writes to `out/` is therefore saved on the Hub, and is kept after the machine is deleted at the end of the job.
3. It installs the packages of `requirements.txt` and runs the file with the same command as on your machine.

The job starts in the background, and the script prints its ID. Follow it with `hf jobs logs -f <job id>`, or on the job page at huggingface.co.

`run_on_hf.sh` contains nothing specific to this project, and can be copied as it is to another one. The bucket is named after the folder. The code must save its results to `out/` (or you change the `OUT` line of the script), and it must download its data from the Hub or from a URL, because only the code is sent to the job.

## Use the trained model

```bash
hf sync hf://buckets/<you>/nanogpt-jobs/out ./out
uv run --with-requirements requirements.txt sample.py "Once upon a time"
```
