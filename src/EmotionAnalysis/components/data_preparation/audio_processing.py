import os
from moviepy.editor import VideoFileClip

from EmotionAnalysis.utils.common_utils import save_to_json


def extract_audio_from_video(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False


def create_audio_dataset(video_data, output_folder_base):
    audio_data = {}
    
    for dataset_type, videos in video_data.items():
        output_folder = os.path.join(output_folder_base, f"audio_{dataset_type}")
        os.makedirs(output_folder, exist_ok=True)
        audio_data[dataset_type] = []

        for video_entry in videos:
            for key, video_path in video_entry.items():
                if key in ["y", "label"]:
                    continue
                
                audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".wav"
                output_path = os.path.join(output_folder, audio_filename)
                
                if extract_audio_from_video(video_path, output_path):
                    audio_data[dataset_type].append({
                        key: output_path,
                        "y": video_entry["y"],
                        "label": video_entry["label"]
                    })
    return audio_data