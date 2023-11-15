from PatientConditonClassification.config.configuration import ConfigurationManager
from PatientConditonClassification.components.data_transformation import DataTransformation
from PatientConditonClassification.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(batch=32, config=data_transformation_config)
        data_transformation.save_transformed_data()