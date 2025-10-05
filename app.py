from flask import Flask, render_template, request, jsonify
import subprocess
import json
import sys
import os
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Print received data for debugging
        print("\nReceived request data:", data, file=sys.stderr)
        
        # Validate required fields
        required_fields = ['rank', 'category', 'location', 'branch', 'topn']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Prepare the command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "scripts/kcet_l2r_train_infer.py",
            "recommend",
            "--rank", str(data['rank']),
            "--category", data['category'],
            "--location", data['location'].upper(),
            "--branch", data['branch'].upper().replace(" ", "_"),
            "--topn", str(data['topn'])
        ]
        
        print("\nExecuting command:", ' '.join(cmd), file=sys.stderr)
        
        # Run the command with a timeout
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=30  # 30 seconds timeout
            )
            
            print("\nCommand output:\n", result.stdout, file=sys.stderr)
            print("\nCommand error (if any):\n", result.stderr, file=sys.stderr)
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout)
                    print(f"\nRaw response data: {response_data}", file=sys.stderr)
                    
                    # Handle different response structures
                    if isinstance(response_data, dict):
                        # If it's a dict with 'data' key
                        if 'data' in response_data and isinstance(response_data['data'], list):
                            recommendations = response_data['data']
                        # If it's a dict with direct array values
                        elif any(isinstance(v, list) for v in response_data.values()):
                            for v in response_data.values():
                                if isinstance(v, list):
                                    recommendations = v
                                    break
                            else:
                                recommendations = []
                        else:
                            recommendations = [response_data]  # Single result as dict
                    elif isinstance(response_data, list):
                        recommendations = response_data
                    else:
                        recommendations = []
                    
                    print(f"\nSuccessfully got {len(recommendations)} recommendations", file=sys.stderr)
                    # Return the data in the expected format
                    return jsonify({
                        "success": True,
                        "data": recommendations
                    })
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse recommendations: {str(e)}\nRaw output: {result.stdout[:500]}"
                    print(f"\n{error_msg}", file=sys.stderr)
                    return jsonify({
                        "success": False,
                        "error": error_msg
                    }), 500
            else:
                error_msg = f"Prediction script failed with code {result.returncode}\nError: {result.stderr}"
                print(f"\n{error_msg}", file=sys.stderr)
                return jsonify({
                    "success": False,
                    "error": error_msg
                }), 500
                
        except subprocess.TimeoutExpired:
            error_msg = "Prediction timed out after 30 seconds"
            print(f"\n{error_msg}", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n\n{traceback.format_exc()}"
        print(f"\n{error_msg}", file=sys.stderr)
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Print Python and environment information
    print(f"\nPython executable: {sys.executable}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)
    print("\nEnvironment variables:", file=sys.stderr)
    for key in sorted(os.environ):
        if 'PATH' in key or 'PYTHON' in key or 'VIRTUAL' in key:
            print(f"{key}: {os.environ[key]}", file=sys.stderr)
    
    # Start the Flask app
    app.run(debug=True, port=5000)
