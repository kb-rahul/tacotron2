import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from io import BytesIO
import wave
from pydub import AudioSegment
import click

hparams = create_hparams()
hparams.sampling_rate = 22050

def load_taco2(chk_pt_path):
    model = load_model(hparams)
    model.load_state_dict(torch.load(chk_pt_path)['state_dict'])
    _ = model.cuda().eval().half()
    return model

def load_waveglow(chk_pt_path):
    waveglow = torch.load(chk_pt_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser

def mel_gen(text, model):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return mel_outputs_postnet

def gen_audio_vocoder(mel_output, waveglow, denoiser):
    with torch.no_grad():
        audio = waveglow.infer(mel_output, sigma=0.666)
    audio = audio/torch.max(torch.abs(audio))
    audio_denoised = denoiser(audio , strength=0.01)[:, 0]
    audio_denoised = audio_denoised/torch.max(torch.abs(audio_denoised))
    return wav_bytes(float2pcm(audio_denoised[0].data.cpu().numpy()))

def gen_e2e_taco(text, taco_model, wavenet_model, denoiser):
    mel_out = mel_gen(text, taco_model)
    return gen_audio_vocoder(mel_out, wavenet_model, denoiser)

def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def wav_bytes(tts_bytes):
    wav_file = BytesIO()
    with wave.open(wav_file, "wb") as obj:
        obj.setnchannels(1)  # mono
        obj.setsampwidth(2)
        obj.setframerate(22050)
        obj.writeframesraw(tts_bytes)
    return wav_file.getvalue()

@click.command()
@click.option('--taco', help='Path to Taco 2 model')
@click.option('--wavenet', help='path to wavenet model')
def main(taco, wavenet):
    taco_model = load_taco2(taco)
    wave_model, denoiser = load_waveglow(wavenet)
    for i in range(5):
        entered_inp = input("Enter your content: ")
        audio_samples = gen_e2e_taco(entered_inp, taco_model, wave_model, denoiser) 
        with open('./test-{}.wav'.format(i), 'wb') as fp:
            fp.write(audio_samples)

if __name__ == "__main__":
    main()
