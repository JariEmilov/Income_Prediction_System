#!/usr/bin/env python
import sys
import click
from income_prediction_agent.crew import IncomePredictionAgent
from dotenv import load_dotenv
load_dotenv()

def run():
    """
    Run the crew.
    """
    print("## Welcome to Income Prediction Agent")
    print('-------------------------------')
    
    # Get user input
    user_query = input("Please describe your demographic information: ")
    
    inputs = {
        'user_query': user_query
    }
    
    try:
        result = IncomePredictionAgent().crew().kickoff(inputs=inputs)
        print("\n## Income Prediction Result:")
        print('-------------------------------')
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def train():
    """
    Train the crew for a given number of iterations.
    """
    print("## Training Income Prediction Agent")
    print('-------------------------------')
    
    inputs = {
        'user_query': "Sample training query: 35 year old married male, college graduate, works in tech"
    }
    
    try:
        IncomePredictionAgent().crew().train(n_iterations=int(sys.argv[1]) if len(sys.argv) > 1 else 1, inputs=inputs)
        print("Training completed!")
    except Exception as e:
        print(f"Training error: {e}")
        sys.exit(1)

def replay():
    """
    Replay the crew execution from a specific task.
    """
    print("## Replaying Income Prediction Agent")
    print('-------------------------------')
    
    try:
        IncomePredictionAgent().crew().replay(task_id=sys.argv[1] if len(sys.argv) > 1 else None)
    except Exception as e:
        print(f"Replay error: {e}")
        sys.exit(1)

def test():
    """
    Test the crew execution and returns the results.
    """
    print("## Testing Income Prediction Agent")
    print('-------------------------------')
    
    inputs = {
        'user_query': "Test: 40 year old single female, masters degree, manager in finance"
    }
    
    try:
        result = IncomePredictionAgent().crew().test(n_iterations=int(sys.argv[1]) if len(sys.argv) > 1 else 1, inputs=inputs)
        print("Test completed!")
        print(result)
    except Exception as e:
        print(f"Test error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()