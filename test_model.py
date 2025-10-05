import lightgbm as lgb
import json
import os

def test_model_loading():
    try:
        model_path = os.path.join('artifacts', 'model_l2r.txt')
        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {os.path.abspath(model_path)}")
            return False
            
        # Try to load the model
        model = lgb.Booster(model_file=model_path)
        print("Successfully loaded the LightGBM model!")
        
        # Print model information
        print("\nModel information:")
        print(f"Number of trees: {model.num_trees()}")
        print(f"Number of features: {model.num_feature()}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    test_model_loading()
