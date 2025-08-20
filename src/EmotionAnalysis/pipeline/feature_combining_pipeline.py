from EmotionAnalysis.config.configuration import ConfigurationManager
from EmotionAnalysis.components.data_preparation.feature_combiner import FeatureCombiner

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    # Feature combining
    feat_comb_config = config_manager.get_feature_combining_config()
    feature_combiner = FeatureCombiner(feat_comb_config)
    feature_combiner.run()