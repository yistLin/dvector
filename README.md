# D-vector

This is a PyTorch implementation of speaker embedding trained with GE2E loss.
The original paper about GE2E loss could be found here: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

## Usage

You can download the pretrained models from: [Wiki - Pretrained Models](https://github.com/yistLin/dvector/wiki/Pretrained-Models).

Since the models are compiled with TorchScript, you can simply load and use a pretrained d-vector anywhere.

```python
import torch
import torchaudio

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)
emb_tensor = dvector.embed_utterance(mel_tensor)
```

## Train from scratch

### Preprocess training data

To use the script provided here, you have to organize your raw data in this way:

- all utterances from a speaker should be put under a directory (**speaker directory**)
- all speaker directories should be put under a directory (**root directory**)
- **speaker directory** can have subdirectories and utterances can be placed under subdirectories

And you can extract utterances from multiple **root directories**, e.g.

```bash
python preprocess.py VoxCeleb1/dev LibriSpeech/train-clean-360 -o preprocessed
```

If you need to modify some audio preprocessing hyperparameters, directly modify `data/wav2mel.py`.
After preprocessing, 3 modules will be saved in the output directory:
1. `wav2mel.pt`
2. `sox_effects.pt`
3. `log_melspectrogram.pt`

> The first module `wav2mel.pt` is actually composed of the second and the third modules.
> These modules were compiled with TorchScript and can be used anywhere to preprocess audio data.

### Train a model

You have to specify where to store checkpoints and logs, e.g.

```bash
python train.py preprocessed <model_dir>
```

During training, logs will be put under `<model_dir>/logs` and checkpoints will be placed under `<model_dir>/checkpoints`.
For more details, check the usage with `python train.py -h`.

### Visualize speaker embeddings

You can visualize speaker embeddings using a trained d-vector.
Note that you have to structure speakers' directories in the same way as for preprocessing.
e.g.

```bash
python visualize.py LibriSpeech/dev-clean -w wav2mel.pt -c dvector.pt -o tsne.jpg
```

The following plot is the dimension reduction result (using t-SNE) of some utterances from LibriSpeech.

![TSNE result](images/tsne.png)

## References

- GE2E-Loss module: [cvqluu/GE2E-Loss](https://github.com/cvqluu/GE2E-Loss)
