# Voice Activity Detection (VAD) Model from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based Voice Activity Detection (VAD) system implemented from scratch using Bidirectional LSTM (BiLSTM) networks. This project demonstrates the application of modern AI techniques in speech processing, focusing on detecting speech and non-speech regions in audio signals, particularly in noisy environments.

## 🚀 Features

- **Deep Learning Architecture**: Utilizes BiLSTM for temporal modeling of speech patterns
- **Noise Robustness**: Trained with noise augmentation to handle low SNR conditions
- **Real-time Capable**: Optimized for frame-level predictions suitable for streaming audio
- **Open Source**: Built with PyTorch and publicly available datasets
- **Educational**: Comprehensive implementation with detailed explanations

## 📋 Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Jupyter Notebook or JupyterLab

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hasithdd/VAD-Model-from-Scratch.git
   cd VAD-Model-from-Scratch
   ```

2. **Install dependencies with uv**:
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all required packages as specified in `pyproject.toml`.

3. **Activate the environment** (optional, uv handles this automatically):
   ```bash
   uv run python --version
   ```

## 📊 Dataset

This project uses the [Google Speech Commands Dataset v0.01](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz), a public dataset containing:

- 65,000 one-second audio clips of 30 different words
- Background noise samples
- Mono WAV files at 16 kHz sample rate

The dataset is automatically downloaded and prepared during notebook execution.

## 🏗️ Model Architecture

The VAD system employs:

- **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs)
- **Network**: Bidirectional LSTM with 2 layers and 128 hidden units
- **Output**: Frame-level binary classification (speech/non-speech)
- **Sequence Length**: 8-second overlapping windows

## 🚀 Usage

1. **Start Jupyter Notebook**:
   ```bash
   uv run jupyter notebook
   ```

2. **Open the main notebook**:
   Navigate to `Cw1_w1987535_HasithVikasithaDharmarathna.ipynb` and open it.

3. **Run the cells sequentially**:
   - The notebook includes data preparation, model training, and evaluation
   - Training takes approximately 10 epochs on a GPU-enabled system
   - Evaluation provides confusion matrix and performance metrics

4. **Key sections**:
   - Data loading and preprocessing
   - Feature extraction (MFCC)
   - Model training with BiLSTM
   - Validation and performance analysis

## 📈 Results

The model achieves:
- **Accuracy**: 86.3%
- **Precision**: 86.1%
- **Recall**: 82.7%
- **F1 Score**: 84.4%

Evaluated on noisy validation data with -10 dB SNR.

## 🔧 Development

### Adding Dependencies

To add new packages:
```bash
uv add package-name
```

### Running Tests

```bash
uv run python -m pytest
```

### Exporting Model

For deployment, the trained model can be exported to ONNX:
```bash
uv run python export_model.py  # (if you create this script)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Google Speech Commands Dataset](https://arxiv.org/abs/1703.05390)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchaudio](https://pytorch.org/audio/stable/index.html)

## 🙏 Acknowledgments

- Based on the coursework for 6COSC020W Applied AI
- Inspired by modern VAD systems like Silero VAD and pyannote.audio

---

**Author**: P.A. Hasith Vikasitha Dharmarathna  
**ID**: 20223265  
**GitHub**: [hasithdd](https://github.com/hasithdd)