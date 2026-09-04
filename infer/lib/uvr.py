import os
from audio_separator.separator import Separator

def separate_vox(input_path, output_dir=None):
    """
    Separate audio using UVR models.
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory for separated files
    
    Returns:
        Tuple of (vocals, instrumental, lead_vocals, backing_vocals, vocals_no_reverb, vocals_reverb)
    """
    
    separator = Separator(output_dir=output_dir)
    
    # Vocals and Instrumental
    vocals = os.path.join(output_dir, 'Vocals.wav')
    instrumental = os.path.join(output_dir, 'Instrumental.wav')

    # Vocals with Reverb and Vocals without Reverb
    vocals_reverb = os.path.join(output_dir, 'Vocals (Reverb).wav')
    vocals_no_reverb = os.path.join(output_dir, 'Vocals (No Reverb).wav')
    
    # Lead Vocals and Backing Vocals
    lead_vocals = os.path.join(output_dir, 'Lead Vocals.wav')
    backing_vocals = os.path.join(output_dir, 'Backing Vocals.wav')

    # Splitting a track into Vocal and Instrumental
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    voc_inst = separator.separate(input_path)
    
    os.rename(os.path.join(output_dir, voc_inst[0]), instrumental)
    os.rename(os.path.join(output_dir, voc_inst[1]), vocals)
    
    # Applying DeEcho-DeReverb to Vocals
    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    voc_no_reverb = separator.separate(vocals)
    os.rename(os.path.join(output_dir, voc_no_reverb[0]), vocals_no_reverb)
    os.rename(os.path.join(output_dir, voc_no_reverb[1]), vocals_reverb)

    # Separating Back Vocals from Main Vocals
    separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    backing_voc = separator.separate(vocals_no_reverb)
    os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals)
    os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals)

    return vocals, instrumental, lead_vocals, backing_vocals, vocals_no_reverb, vocals_reverb
