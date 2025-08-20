from EmotionAnalysis.components.trainer.video_frames_extraction import Video_Feature_Extraction_Trainer

if __name__ == "__main__":
    config_path = "config/paths.yaml"
    params_path = "config/params.yaml"
    
    video_trainer = Video_Feature_Extraction_Trainer(config_path, params_path)
    video_trainer.run_pipeline()