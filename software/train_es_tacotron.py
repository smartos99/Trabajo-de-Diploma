import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# from TTS.tts.datasets.tokenizer import Tokenizer

#output_path = os.path.dirname(os.path.abspath(__file__))
output_path = "/content/drive/MyDrive/Thesis/Tacotron2/traineroutput/"

# init configs
dataset_config = BaseDatasetConfig(
    name="mailabs", meta_file_train="metadata.csv", path=os.path.join(output_path, "/content/drive/MyDrive/Thesis/sandra_voice/by_book/female/sandra/sandrav/")
)

audio_config = BaseAudioConfig(
    sample_rate=16000,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=True,
    mel_fmin=50.0,
    mel_fmax= 7600,
    spec_gain=1.0,
    log_func="np.log10",
    ref_level_db=20,
    preemphasis=0.0,
)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    run_name="tacotron2_es_run_name",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=0,
    run_eval=True,
    test_delay_epochs=10,
    r=7,
    gradual_training=[[0, 7, 64], [1, 5, 64], [50000, 3, 32], [130000, 2, 32], [290000,1,32]],
    double_decoder_consistency=True,
    epochs=2000,
    i_epochs=212,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="es",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),   ########
    precompute_num_workers=0,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
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

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
