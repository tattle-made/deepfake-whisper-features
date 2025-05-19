import argparse
import yaml
import torch
import torchaudio
import os
from src.models import models

SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480_000  # <- originally 64_600


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

def apply_preprocessing(
    waveform,
    sample_rate,
):
    if sample_rate != SAMPLING_RATE and SAMPLING_RATE != -1:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    # Stereo to mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    # Trim too long utterances...
    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    # ... or pad too short ones.
    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def resample_file(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform



def load_data(path):
    waveform, sample_rate = torchaudio.load(path, normalize=APPLY_NORMALIZATION)

    waveform, sample_rate = apply_preprocessing(waveform, sample_rate)

    return [waveform, sample_rate]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )
    parser.add_argument('--input_path', type=str,default=None, help='Change this to actual path to the audio file')
    parser.add_argument('--model_path', type=str,default=None, help='Model checkpoint')

    args = parser.parse_args()

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    device = 'cpu'
    print('Device: {}'.format(device))

    with open(args.config, "r") as f:
        model_config = yaml.safe_load(f)


    model_name, model_parameters = model_config['model']["name"], model_config['model']["parameters"]
    
    # Load model architecture
    model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if len(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    input_path = args.input_path

    file_paths = os.listdir(input_path)
    final_paths = [os.path.join(input_path, i) for i in file_paths]
    
    batch_x = []

    for file_path in final_paths:
        waveform, _ = load_data(file_path)
        waveform = waveform
        batch_x.append(waveform)

    batch_x = torch.stack(batch_x).to(device)

    # Run prediction
    with torch.no_grad():
        pred = model(batch_x) # adjust depending on model output shape
        pred = torch.sigmoid(pred)
        batch_pred_label = (pred + 0.5).int()
        result_array = (batch_pred_label.squeeze() == 0)  # 0 → True, 1 → False

    print(result_array.cpu().numpy())