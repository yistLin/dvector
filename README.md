# D-vector

This is a PyTorch implementation of speaker embedding trained with GE2E loss.
The original paper about GE2E loss could be found here: [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

## Usage

```python
import torch
import torchaudio

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
```

You can also embed multiple utterances of a speaker at once:

```python
emb_tensor = dvector.embed_utterances([mel_tensor_1, mel_tensor_2])  # shape: (emb_dim)
```

There are 2 modules in this example:
- `wav2mel.pt` is the preprocessing module which is composed of 2 modules:
    - `sox_effects.pt` is used to normalize volume, remove silence, resample audio to 16 KHz, 16 bits, and remix all channels to single channel
    - `log_melspectrogram.pt` is used to transform waveforms to log mel spectrograms
- `dvector.pt` is the speaker encoder

Since all the modules are compiled with [TorchScript](https://pytorch.org/docs/stable/jit.html), you can simply load them and use anywhere **without any dependencies**.

### Pretrianed models & preprocessing modules

You can download them from the page of [*Releases*](https://github.com/yistLin/dvector/releases).

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
After preprocessing, 3 preprocessing modules will be saved in the output directory:
1. `wav2mel.pt`
2. `sox_effects.pt`
3. `log_melspectrogram.pt`

> The first module `wav2mel.pt` is composed of the second and the third modules.
> These modules were compiled with TorchScript and can be used anywhere to preprocess audio data.

### Train a model

You have to specify where to store checkpoints and logs, e.g.

```bash
python train.py preprocessed <model_dir>
```

During training, logs will be put under `<model_dir>/logs` and checkpoints will be placed under `<model_dir>/checkpoints`.
For more details, check the usage with `python train.py -h`.

### Use different speaker encoders

By default I'm using 3-layerd LSTM with attentive pooling as the speaker encoder, but you can use speaker encoders of different architecture.
For more information, please take a look at `modules/dvector.py`.

## Visualize speaker embeddings

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
