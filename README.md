# D-Vector

This is the PyTorch implementation of speaker embedding (d-vector) trained with GE2E loss.

The original paper about GE2E loss could be found here: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

## Usage

### Prepare training data

To use the script provided here, you have to organize your raw data in this way:

- all utterances from a speaker should be put under a directory (**speaker directory**)
- all speaker directories should be put under a directory (**root directory**)
- **speaker directory** can have subdirectories and utterances can be placed under subdirectories

You have to specify two things here:

- use `-e` or `--extensions` to specify the extensions of utterances to be extracted and separate them with commas e.g. `wav,flac,mp3` (do not leave **SPACES** in between)
- use `-s` or `--save_dir` to specify the directory for saving processed utterances

And a good thing about this script is that you can extract utterances from multiple **root directories**.
For example:

```bash
python prepare.py -s data -e wav,flac VCTK-Corpus/wav48 LibriSpeech/train-clean-360
```

### Start training

Only `DATA_DIR` and `MODEL_DIR` have to be specified here.
For more details, check the usage with `python train.py -h`.

```bash
python train.py DATA_DIR MODEL_DIR \
    -i 1000000 \
    -s 10000 \
    -d 100000 \
    -n 64 \
    -m 10 \
    -l 128
```

During training, event logs will be put under `MODEL_DIR`.

### Continue training from saved checkpoints

To continue the training from a saved checkpoint, just specify the checkpoint path with `-c` or `--checkpoint_path`.
Note that you can still specify other optional arguments because they might be different from the ones in the previous training.

```bash
python train.py DATA_DIR MODEL_DIR \
    -i 500000 \
    -s 100000 \
    -d 100000 \
    -n 64 \
    -m 10 \
    -l 128 \
    -c SAVED_CHECKPOINT
```

## Results

The dimension reduction result (using t-SNE) of some utterances from LibriSpeech.

![TSNE result](images/tsne.png)

### Credits

The GE2E-Loss module is first borrowed from [cvqluu/GE2E-Loss](https://github.com/cvqluu/GE2E-Loss) and then rewritten and optimized for speed by myself.
