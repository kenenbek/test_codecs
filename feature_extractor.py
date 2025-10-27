from transformers import MimiModel, AutoFeatureExtractor


feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")


print(feature_extractor)