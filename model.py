import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Define parameters
image_size = (224, 224)  # Resize images to 224x224 pixels
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,       # Normalize pixel values to [0,1]
    shear_range=0.2,       # Randomly shear images
    zoom_range=0.2,        # Randomly zoom images
    horizontal_flip=True   # Randomly flip images
)

val_datagen = ImageDataGenerator(rescale=1.0/255)  # Only normalize for validation

# Load images
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Allen\Desktop\Food Recipe model',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = val_datagen.flow_from_directory(
    r'C:\Users\Allen\Desktop\Food Recipe model',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
# Load pre-trained ResNet50 model (excluding the top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build your model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Average pooling layer to reduce dimensions
    Dense(512, activation='relu'),  # Fully connected layer
    Dense(train_generator.num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use raw string to avoid escape sequences
file_path = r'C:\Users\Allen\Desktop\Food Recipe model\Recipe Data\Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
recipe_df = pd.read_csv(file_path)
print(recipe_df.head())

def get_recipes_for_title(Title, recipe_df):
    """
    Retrieve recipes for a given food Title.
    
    Parameters:
        Title (str): The food title identified from the image.
        recipe_df (pd.DataFrame): The dataframe containing the recipes.
        
    Returns:
        list: A list of recipes (as dictionaries) for the given Title.
    """
    recipes = recipe_df[recipe_df['Title'] == Title]
    return recipes.to_dict('records')

# Simulate a predicted food Title
predicted_title = 'Drunk Apricot Shito (Ghanaian Hot Pepper Sauce)'  # Replace with actual prediction

# Fetch recipes for the predicted Title
recommended_recipes = get_recipes_for_title(predicted_title, recipe_df)

# Display the recommended recipes
for recipe in recommended_recipes:
    print(f"Recipe ID: {recipe['Image_Name']}")
    print(f"Ingredients: {recipe['Ingredients']}")
    print(f"Instructions: {recipe['Instructions']}")
    print("-" * 40)



history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # Number of epochs to train
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
