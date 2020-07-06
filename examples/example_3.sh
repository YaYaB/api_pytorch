# Create service 1
curl -X PUT "localhost:8000/create" -d '{"model_name": "resnet18", "service_name": "s1", "type":"image"}'

# Create service 2
curl -X PUT "localhost:8000/create" -d '{"model_name": "wide_resnet101_2", "service_name": "s2", "type":"image"}'

# Make prediction using model 1
curl -X POST "localhost:8000/predict" -d '{"service_name": "s1", "data":["https://images.daznservices.com/di/library/GOAL/99/e4/cristiano-ronaldo-juventus-2019-20_16jgivl5d7ean1cw4aymindbiy.jpg", "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg"]}'

# Make prediction model 2
curl -X POST "localhost:8000/predict" -d '{"service_name": "s2", "data":["https://images.daznservices.com/di/library/GOAL/99/e4/cristiano-ronaldo-juventus-2019-20_16jgivl5d7ean1cw4aymindbiy.jpg", "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg"]}'
