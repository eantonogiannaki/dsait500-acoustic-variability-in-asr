from variant_factory.AcousticMetricsCalculator import AcousticMetricsCalculator
from variant_factory.DatasetVariantBuilder import DatasetVariantBuilder
from variant_factory.UtteranceBalancer import UtteranceBalancer


class DatasetVariantFactory:
    """
    Owns a single FeatureSchema and an AcousticMetricsCalculator built from it, and builds
    UtteranceBalancer / DatasetVariantBuilder instances on demand — so every helper it
    creates is guaranteed to share the same column-name schema.
    """

    def __init__(self, schema, metrics=None):
        self.schema = schema
        self.acoustics_calculator = AcousticMetricsCalculator(schema=schema, metrics=metrics)

    def create_utterance_balancer(self, seconds_per_speaker):
        return UtteranceBalancer(schema=self.schema, seconds_per_speaker=seconds_per_speaker)

    def create_dataset_builder(self, num_speakers, metric, training_features, scaler=None, normalization=False):
        return DatasetVariantBuilder(
            schema=self.schema, num_speakers=num_speakers, metric=metric,
            calculator=self.acoustics_calculator, training_features=training_features,
            scaler=scaler, normalization=normalization
        )
