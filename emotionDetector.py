from config import *
import random

# Input: 48*48 image (grayscale), landmark
# Output: emotion (Returning 0 means failure to find emotion)
def get_emotion(image, landmark):
    # Implement emotion detection algorithm here
    # For now, return random emotion
    return random.choice(emotion + [0])