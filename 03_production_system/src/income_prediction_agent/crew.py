from crewai import Agent, Crew, Process, Task
from typing import List
import os
import re
from income_prediction_agent.tools.custom_tool import income_prediction_tool

class IncomePredictionAgent():
    """RAG-Enhanced Income Prediction Agent using Ensemble ML"""
    
    def __init__(self):
        print("🏗️ Initializing Income Prediction Agent...")
        
        # Test the custom tool immediately
        print("🔧 Testing custom tool...")
        try:
            # Test with proper input format
            test_result = income_prediction_tool._run("35 year old male software engineer with masters degree")
            print(f"✅ Custom tool test successful: {test_result[:100]}...")
        except Exception as e:
            print(f"❌ Custom tool test failed: {e}")
            import traceback
            traceback.print_exc()

    def income_predictor(self) -> Agent:
        """Income prediction agent with access to the ML tool"""
        return Agent(
            role='Senior Data Scientist',
            goal='Execute income predictions using the income_prediction_tool and provide comprehensive analysis',
            backstory='''You are a senior data scientist who specializes in income prediction. Your job is to EXECUTE the income_prediction_tool
            with user demographic data and present the results clearly. You MUST call the tool - do not just plan to call it.
            You are results-oriented and always complete the prediction task by actually using your tools.''',
            tools=[income_prediction_tool],
            verbose=True,
            allow_delegation=False,
            max_execution_time=300,
            max_retry_limit=3
        )

    def predict_income_task(self) -> Task:
        """Task for predicting income using the agent"""
        return Task(
            description='''EXECUTE the income_prediction_tool immediately with this input: "{user_input}"

            You MUST:
            1. Call income_prediction_tool(query="{user_input}") RIGHT NOW
            2. Wait for the tool response
            3. Present the complete results
            
            DO NOT just say you will call the tool - ACTUALLY CALL IT NOW.
            The user is waiting for the prediction results.''',
            
            expected_output='''The complete output from the income_prediction_tool including:
            - Ensemble prediction with probability percentage
            - Similar profiles analysis from RAG
            - Demographic breakdown
            - Technical methodology details
            
            Present the tool output exactly as received with clear formatting.''',
            
            agent=self.income_predictor()
        )

    def crew(self) -> Crew:
        """Creates the income prediction crew"""
        return Crew(
            agents=[self.income_predictor()],
            tasks=[self.predict_income_task()],
            process=Process.sequential,
            verbose=True,
            max_execution_time=600  # 10 minutes total timeout
        )

    def predict(self, user_input: str) -> str:
        """Run the crew to get actual agent-based prediction"""
        try:
            print(f"🚀 Starting CrewAI prediction for: {user_input}")
            
            # Test tool before running crew
            print("🔧 Pre-flight tool check...")
            try:
                tool_test = income_prediction_tool._run(user_input)
                print(f"✅ Tool working - preview: {tool_test[:200]}...")
            except Exception as tool_error:
                print(f"⚠️ Tool test failed: {tool_error}")
                return f"Tool test failed: {tool_error}"
            
            # Create crew and kickoff
            my_crew = self.crew()
            print(f"🤖 Crew created with {len(my_crew.agents)} agents")
            print(f"📋 Crew has {len(my_crew.tasks)} tasks")
            
            # Force execution with shorter timeout
            result = my_crew.kickoff(
                inputs={'user_input': user_input},
            )
            print(f"✅ CrewAI completed successfully")
            return str(result)
            
        except Exception as e:
            print(f"❌ CrewAI Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try direct tool call as fallback
            print("🔄 Attempting direct tool fallback...")
            try:
                direct_result = income_prediction_tool._run(user_input)
                return f"DIRECT TOOL RESULT (CrewAI failed):\n\n{direct_result}"
            except Exception as tool_error:
                return f"Both Crew and Tool failed. CrewAI Error: {str(e)}, Tool Error: {str(tool_error)}"

    def direct_predict(self, user_input: str) -> str:
        """Direct prediction method - ONLY as fallback"""
        try:
            print("🔧 Using direct tool access...")
            result = income_prediction_tool._run(user_input)
            return result
        except Exception as e:
            print(f"❌ Direct tool error: {e}")
            import traceback
            traceback.print_exc()
            return f"Direct prediction error: {str(e)}"