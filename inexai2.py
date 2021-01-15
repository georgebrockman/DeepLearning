from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory('/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI')
model_trainer.trainModel(num_objects=2, num_experiments=100, enhance_data=True, batch_size=64, show_network_summary=True)
