import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

#output_path = os.path.dirname(os.path.abspath(__file__))
output_path = "/content/drive/MyDrive/Thesis/VITS/traineroutput/"

dataset_config = BaseDatasetConfig(
    name="mailabs", meta_file_train="metadata.csv", path=os.path.join(output_path, "/content/drive/MyDrive/Thesis/sandra_voice/by_book/female/sandra/sandrav/")
)
audio_config = VitsAudioConfig(
    sample_rate=22050,
     win_length=1024, 
     hop_length=256, 
     num_mels=80, 
     mel_fmin=0,
    mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_spanish",
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="es",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    test_sentences=[
        "Hola, mi nombre es Sandra.",
        "Diseñamos toda clase de casas.",
        "Mis secretos obstáculos.",
        "número 70",
        "¡Ninguna persona mayor comprenderá jamás que esto sea verdaderamente importante!",
    ],
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()