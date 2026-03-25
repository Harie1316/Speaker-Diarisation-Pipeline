# Speaker Diarisation Pipeline

An end-to-end speaker diarisation system that identifies *who spoke when* in multi-speaker audio, combining neural voice activity detection, speaker embedding extraction, agglomerative clustering, and ASR transcription into a unified pipeline.

Evaluated on the **AMI Meeting Corpus** — a standard benchmark for meeting speech analysis.

## Overview

The pipeline takes a raw meeting recording as input and produces a **speaker-attributed transcript** as output. It follows the standard diarisation architecture:

```
Raw Audio → VAD → Speaker Embeddings → Clustering → Diarisation → ASR → Attributed Transcript
```

Each stage is implemented as a separate, inspectable module:

| Stage | Model / Method | Output |
|-------|---------------|--------|
| **Voice Activity Detection** | pyannote/segmentation-3.0 (PyanNet) | Binary speech/non-speech timeline |
| **Speaker Embedding** | ECAPA-TDNN (512-d d-vectors) | Fixed-dimensional speaker representations |
| **Clustering** | Agglomerative (average linkage, cosine distance) | Speaker cluster assignments |
| **End-to-End Diarisation** | pyannote/speaker-diarization-3.1 | Speaker-labelled segmentation (RTTM) |
| **ASR Transcription** | OpenAI Whisper (small.en) | Timestamped word-level transcription |
| **Attribution** | Temporal alignment (majority vote) | Speaker-labelled transcript |

## Results

Evaluated on the AMI Meeting Corpus recording IS1000a (Mix-Headset):

### Voice Activity Detection

| Metric | Value |
|--------|:---:|
| Speech segments detected | 11 |
| Total speech duration | 19.91s |
| Speech ratio | 33.2% |
| Mean segment duration | 1.81s |

### Speaker Diarisation

| Metric | Value |
|--------|:---:|
| Speakers detected | 3 |
| Speaker turns (60s excerpt) | 17 |
| Embedding dimensionality | 512-d (ECAPA-TDNN) |
| Clustering | Agglomerative (average linkage, cosine) |

### Diarisation Error Rate (DER)

| Component | Value |
|-----------|:---:|
| **DER (total)** | **25.46%** |
| Miss | 11.77% |
| False Alarm | 7.22% |
| Speaker Confusion | 6.46% |
| Reference speech total | 972.1s |
| Collar | 0.25s |

> DER is the primary evaluation metric for speaker diarisation, defined as the fraction of reference speech time that is incorrectly labelled: DER = Miss + False Alarm + Speaker Confusion.

### Sample Attributed Transcript

```
[SPEAKER_04]  00:30 – 00:38
  I think it is supposed to be like this. Okay.

[SPEAKER_01]  00:38 – 00:40
  ...

[SPEAKER_00]  00:40 – 00:55
  ...
```

## Pipeline Architecture

### 1. Voice Activity Detection
The pyannote VAD model (PyanNet segmentation network) operates on 10ms frames to produce a binary speech/non-speech timeline. Only speech segments are passed downstream, gating all subsequent processing.

### 2. Speaker Embedding Extraction
Each speech segment is passed through a pre-trained ECAPA-TDNN speaker encoder which maps variable-length audio to a fixed 512-dimensional d-vector. These embeddings capture speaker-discriminative characteristics — vocal tract geometry, pitch statistics, and prosody.

### 3. Agglomerative Clustering
Speaker embeddings are grouped via agglomerative clustering with average linkage and cosine distance. A threshold of 0.65 determines when to stop merging clusters, automatically estimating the number of speakers.

### 4. End-to-End Diarisation
The full pyannote/speaker-diarization-3.1 pipeline combines all stages (segmentation → embedding → clustering) into a single pass, producing RTTM-formatted output with speaker labels and timestamps.

### 5. ASR + Speaker Attribution
OpenAI Whisper (small.en) transcribes the audio with word-level timestamps. Each ASR segment is then aligned with the diarisation output via majority-vote temporal overlap, assigning a speaker label to each transcribed utterance.

## Dataset

**AMI Meeting Corpus** — a multi-modal corpus of real meeting recordings widely used as a benchmark for speaker diarisation and meeting analysis research. The pipeline uses the IS1000a recording (Mix-Headset channel, 16kHz mono).

## Project Structure

```
├── Speaker_Diarisation_Pipeline.ipynb   # Complete pipeline notebook
├── README.md
└── pipeline_summary.png                 # Generated summary figure
```

## How to Run

1. Open `Speaker_Diarisation_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. You will need a **HuggingFace access token** with access to:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Accept the model terms on both model pages before running
4. Run all cells sequentially
5. Total runtime: ~5–10 minutes on a T4 GPU

## Tech Stack

- **Python** — core language
- **PyTorch** — neural network inference
- **pyannote.audio** — speaker diarisation pipeline (VAD, segmentation, embedding, clustering)
- **pyannote.metrics** — DER evaluation
- **OpenAI Whisper** — automatic speech recognition
- **torchaudio / librosa / soundfile** — audio I/O and processing
- **Matplotlib** — visualisation
