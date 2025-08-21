from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import pandas as pd
import joblib
import pickle
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

class IncomeInput(BaseModel):
    """Input schema for income prediction tool."""
    query: str = Field(..., description="User demographic query describing age, education, occupation, marital status, etc.")

class IncomePredictor:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.load_model_components()
        self.load_rag_data()
        self.setup_rag_system()
    
    def load_model_components(self):
        """Load the trained model and preprocessing components"""
        try:
            # Navigate to the correct path from the tools directory
            base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
            model_path = os.path.join(base_path, 'models', 'production_model.pkl')
            preprocessing_path = os.path.join(base_path, 'models', 'preprocessing_components.pkl')
            
            self.model = joblib.load(model_path)
            
            with open(preprocessing_path, 'rb') as f:
                components = pickle.load(f)
                self.target_encoder = components['target_encoder']
                self.feature_encoders = components['feature_encoders']
                self.feature_columns = components['feature_columns']
                self.categorical_columns = components['categorical_columns']
                self.numerical_columns = components['numerical_columns']
            
            if self.verbose:
                print("✅ Model and preprocessing components loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_rag_data(self):
        """Load training data and metadata for RAG system"""
        try:
            base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
            training_data_path = os.path.join(base_path, 'training_data', 'training_data.csv')
            metadata_path = os.path.join(base_path, 'models', 'production_metadata.json')
            column_info_path = os.path.join(base_path, 'training_data', 'column_info.json')
            education_mapping_path = os.path.join(base_path, 'training_data', 'education_mapping.csv')
            
            self.training_data = pd.read_csv(training_data_path)
            
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            with open(column_info_path, 'r') as f:
                self.column_info = json.load(f)
            
            self.education_mapping = pd.read_csv(education_mapping_path)
            
            if self.verbose:
                print(f"✅ RAG data loaded successfully - Training data shape: {self.training_data.shape}")
                
        except Exception as e:
            print(f"❌ Error loading RAG data: {e}")
            raise
    
    def setup_rag_system(self):
        """Setup RAG system with training data embeddings"""
        try:
            self.training_descriptions = []
            for _, row in self.training_data.iterrows():
                desc = self.create_person_description(row)
                self.training_descriptions.append(desc)
            
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            self.training_vectors = self.vectorizer.fit_transform(self.training_descriptions)
            
            if self.verbose:
                print(f"✅ RAG system setup completed - {len(self.training_descriptions)} training descriptions")
                
        except Exception as e:
            print(f"❌ Error setting up RAG system: {e}")
            raise
    
    def normalize_query_for_rag(self, query):
        """Normalize written numbers and other patterns before RAG search"""
        normalized_query = query.lower()
        
        # Written number mapping
        word_to_num = {
            'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        # Convert written ages to numbers for better RAG matching
        written_age_pattern = r'\b(twenty|thirty|forty|fifty|sixty|seventy)(?:\s+(one|two|three|four|five|six|seven|eight|nine))?\s*(years?\s*old|yr\s*old|year\s*old)\b'
        match = re.search(written_age_pattern, normalized_query)
        if match:
            tens = word_to_num.get(match.group(1), '0')
            ones = word_to_num.get(match.group(2), '0') if match.group(2) else '0'
            age_num = str(int(tens) + int(ones))
            age_suffix = match.group(3)
            replacement = f"{age_num} {age_suffix}"
            normalized_query = re.sub(written_age_pattern, replacement, normalized_query)
        
        # Normalize education terms for better matching
        education_normalizations = {
            r'\bbachelor\'?s?\s*(degree)?\b': 'bachelor degree',
            r'\bmaster\'?s?\s*(degree)?\b': 'master degree',
            r'\bdoctorate?\b': 'doctorate',
            r'\bphd\b': 'doctorate',
            r'\bhigh\s*school\s*(diploma|graduate)?\b': 'high school graduate',
            r'\bassociate\'?s?\s*(degree)?\b': 'associate degree'
        }
        
        for pattern, replacement in education_normalizations.items():
            normalized_query = re.sub(pattern, replacement, normalized_query)
        
        return normalized_query
    
    def create_person_description(self, row):
        """Create a natural language description of a person from training data"""
        age = row['age']
        sex = row['sex']
        
        education_map = {
            1: 'preschool', 2: 'elementary school', 3: 'elementary school', 4: 'middle school',
            5: 'some high school', 6: 'some high school', 7: 'some high school', 8: 'high school graduate',
            9: 'high school graduate', 10: 'some college', 11: 'associate degree', 12: 'associate degree',
            13: 'bachelor degree', 14: 'master degree', 15: 'professional degree', 16: 'doctorate'
        }
        education = education_map.get(row['education_num'], 'high school graduate')
        
        occupation_map = {
            'Prof-specialty': 'professional or specialist',
            'Exec-managerial': 'executive or manager',
            'Sales': 'sales representative',
            'Craft-repair': 'craftsman or repair worker',
            'Machine-op_inspct': 'machine operator',
            'Other': 'service worker'
        }
        occupation = occupation_map.get(row['occupation_grouped'], 'worker')
        
        marital_map = {
            'Never-married': 'single',
            'Married-civ-spouse': 'married',
            'Divorced': 'divorced',
            'Separated': 'separated',
            'Widowed': 'widowed',
            'Married-spouse-absent': 'married',
            'Married-AF-spouse': 'married'
        }
        marital_status = marital_map.get(row['marital_status'], 'single')
        
        workclass_map = {
            'Private': 'private company',
            'Self-emp-not-inc': 'self employed',
            'Self-emp-inc': 'self employed business owner',
            'Federal-gov': 'government',
            'Local-gov': 'government',
            'State-gov': 'government',
            'Without-pay': 'volunteer',
            'Never-worked': 'unemployed'
        }
        workclass = workclass_map.get(row['workclass'], 'private company')
        
        hours = row['hours_per_week']
        description = f"{age} year old {sex.lower()} with {education}, works as {occupation} in {workclass}, {marital_status}"
        
        if hours != 40:
            description += f", works {hours} hours per week"
            
        return description
    
    def find_similar_people(self, user_query, top_k=5):
        """Find similar people using normalized query for better RAG matching"""
        try:
            normalized_query = self.normalize_query_for_rag(user_query)
            query_vector = self.vectorizer.transform([normalized_query])
            similarities = cosine_similarity(query_vector, self.training_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            similar_people = []
            for idx in top_indices:
                person_data = self.training_data.iloc[idx].to_dict()
                person_data['description'] = self.training_descriptions[idx]
                person_data['similarity_score'] = similarities[idx]
                similar_people.append(person_data)
            
            return similar_people
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error in RAG similarity search: {e}")
            return []
    
    def extract_features_from_similar_people(self, similar_people, user_query):
        """Enhanced feature extraction with improved parsing"""
        if not similar_people:
            return self.get_default_features()
        
        base_person = similar_people[0]
        features = base_person.copy()
        query_lower = user_query.lower()
        
        # Enhanced feature extraction
        age = self.extract_age_enhanced(query_lower)
        if age:
            features['age'] = age
        
        gender = self.extract_gender_enhanced(query_lower)
        if gender:
            features['sex'] = gender
        
        education_num = self.extract_education_enhanced(query_lower)
        if education_num:
            features['education_num'] = education_num
        
        marital_info = self.extract_marital_enhanced(query_lower)
        if marital_info:
            features['marital_status'] = marital_info['status']
            features['relationship'] = marital_info['relationship']
        
        occupation = self.extract_occupation_enhanced(query_lower)
        if occupation:
            features['occupation_grouped'] = occupation
        
        hours = self.extract_hours_enhanced(query_lower)
        if hours:
            features['hours_per_week'] = hours
        
        return features
    
    def extract_age_enhanced(self, query_lower):
        """Enhanced age extraction with written numbers"""
        word_to_num = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        }
        
        # Try numeric patterns first
        age_patterns = [
            r'(\d+)[-\s]*year[-\s]*old',
            r'age[:\s]*(\d+)',
            r'(\d+)[:\s]*years?',
            r'\bim\s*(\d+)\b',
            r"i'm\s*(\d+)\b",
            r'\b(\d+)\s*yr\s*old\b'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        # Try written numbers
        written_pattern = r'\b(twenty|thirty|forty|fifty|sixty|seventy)(?:\s+(one|two|three|four|five|six|seven|eight|nine))?\b'
        match = re.search(written_pattern, query_lower)
        if match:
            tens = word_to_num.get(match.group(1), 0)
            ones = word_to_num.get(match.group(2), 0) if match.group(2) else 0
            return tens + ones
        
        return None
    
    def extract_gender_enhanced(self, query_lower):
        """Enhanced gender extraction"""
        if re.search(r'\b(female|woman|lady|girl|she|her|femal|gal)\b', query_lower):
            return 'Female'
        elif re.search(r'\b(male|man|gentleman|guy|he|him|dude|boy)\b', query_lower):
            return 'Male'
        return None
    
    def extract_education_enhanced(self, query_lower):
        """Enhanced education extraction with FIXED mapping"""
        education_patterns = {
            r'\b(phd|doctorate|doctoral|ph\.d)\b': 16,
            r'\b(masters?|master\'?s|msc|mba|ms)\b': 14, 
            r'\b(bachelors?|bachelor\'?s|college|university|ba|bs|b\.a|b\.s)\b': 13,
            r'\b(associates?|associate\'?s|aa|as)\b': 12,
            r'\b(high\s*school|highschool|diploma|ged|hs)\b': 9,
            r'\b(some\s*college|college\s*dropout)\b': 10
        }
        
        for pattern, edu_num in education_patterns.items():
            if re.search(pattern, query_lower):
                return edu_num
        
        return None
    
    def extract_marital_enhanced(self, query_lower):
        """Enhanced marital status extraction"""
        if re.search(r'\b(never\s*married|unmarried|single|bachelor|bachelorette)\b', query_lower):
            return {'status': 'Never-married', 'relationship': 'Not-in-family'}
        elif re.search(r'\b(married|marriage|spouse|husband|wife|wed)\b', query_lower):
            return {'status': 'Married-civ-spouse', 'relationship': 'Husband'}
        elif re.search(r'\b(divorced|separated)\b', query_lower):
            return {'status': 'Divorced', 'relationship': 'Not-in-family'}
        elif re.search(r'\b(widowed|widow|widower)\b', query_lower):
            return {'status': 'Widowed', 'relationship': 'Not-in-family'}
        
        return None
    
    def extract_occupation_enhanced(self, query_lower):
        """Enhanced occupation extraction"""
        occupation_patterns = {
            'Prof-specialty': [
                r'\b(engineer|engineering|data\s*scientist|scientist|analyst|programmer|developer|software|architect|doctor|physician|lawyer|attorney|teacher|professor|instructor|nurse|therapist)\b'
            ],
            'Exec-managerial': [
                r'\b(manager|executive|director|supervisor|admin|ceo|president|management|exec)\b'
            ],
            'Sales': [
                r'\b(sales|salesman|salesperson|retail)\b'
            ],
            'Craft-repair': [
                r'\b(construction|electrician|plumber|mechanic|carpenter|repair|craft)\b'
            ],
            'Other': [
                r'\b(waiter|waitress|server|janitor|cleaner|driver|uber|lyft|service)\b'
            ]
        }
        
        for occupation_group, patterns in occupation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return occupation_group
        
        return None
    
    def extract_hours_enhanced(self, query_lower):
        """Enhanced hours extraction"""
        hours_patterns = [
            r'(\d+)\s*hours?\s*(per\s*week|weekly|\/week|a\s*week)',
            r'(\d+)\s*hrs?\s*(per\s*week|weekly|\/week|a\s*week)',
            r'work\s*(\d+)\s*hours?',
            r'working\s*(\d+)\s*hours?',
            r'(\d+)\+?\s*hours?',
            r'(\d+)\s*hors\b'  # Handle typo "hors"
        ]
        
        for pattern in hours_patterns:
            match = re.search(pattern, query_lower)
            if match:
                hours = int(match.group(1))
                return min(hours, 99)
        
        return None
    
    def get_default_features(self):
        """Get default features if no similar people found"""
        return {
            'age': 39,
            'sex': 'Male',
            'education_num': 9,
            'marital_status': 'Never-married',
            'relationship': 'Not-in-family',
            'race': 'White',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 40,
            'workclass': 'Private',
            'occupation_grouped': 'Prof-specialty',
            'native_country_grouped': 'United-States'
        }
    
    def preprocess_features(self, features):
        """Convert features to model input format"""
        final_df = pd.DataFrame(0, index=[0], columns=self.feature_columns)
        
        # Numerical columns
        for col in self.numerical_columns:
            if col in features:
                final_df[col] = features[col]
        
        # Encode gender
        if 'sex' in features and 'sex' in self.feature_encoders:
            try:
                sex_encoded = self.feature_encoders['sex'].transform([features['sex']])[0]
                final_df['sex'] = sex_encoded
            except ValueError:
                final_df['sex'] = 0
        
        # One-hot encode categorical features
        categorical_mappings = {
            'workclass': features.get('workclass', 'Private'),
            'marital_status': features.get('marital_status', 'Never-married'),
            'relationship': features.get('relationship', 'Not-in-family'),
            'race': features.get('race', 'White'),
        }
        
        for category, value in categorical_mappings.items():
            col_name = f"{category}_{value}"
            if col_name in final_df.columns:
                final_df[col_name] = 1
        
        # Handle grouped features
        occupation_col = f"occupation_grouped_{features.get('occupation_grouped', 'Prof-specialty')}"
        if occupation_col in final_df.columns:
            final_df[occupation_col] = 1
        
        country_col = f"native_country_grouped_{features.get('native_country_grouped', 'United-States')}"
        if country_col in final_df.columns:
            final_df[country_col] = 1
        
        return final_df
    
    def predict_from_query_ensemble(self, query: str) -> str:
        """Enhanced prediction with detailed similar profiles"""
        try:
            similar_people = self.find_similar_people(query, top_k=5)
            
            if not similar_people:
                features = self.get_default_features()
                processed_data = self.preprocess_features(features)
                probability = self.model.predict_proba(processed_data)[0]
                prob_over_50k = probability[1] * 100
                return f"PREDICTION: Limited data available. Estimated {prob_over_50k:.1f}% probability of earning over $50K annually."
            
            # Get predictions from similar people
            similar_predictions = []
            similar_probabilities = []
            
            for person in similar_people:
                processed_data = self.preprocess_features(person)
                prob = self.model.predict_proba(processed_data)[0][1] * 100
                similar_probabilities.append(prob)
            
            # Extract user-specific features
            user_features = self.extract_features_from_similar_people(similar_people, query)
            processed_user_data = self.preprocess_features(user_features)
            user_probability = self.model.predict_proba(processed_user_data)[0][1] * 100
            
            # Create ensemble prediction
            similarity_weights = np.array([person['similarity_score'] for person in similar_people])
            similarity_weights = similarity_weights / similarity_weights.sum()
            
            weighted_similar_prob = np.average(similar_probabilities, weights=similarity_weights)
            ensemble_probability = 0.6 * user_probability + 0.4 * weighted_similar_prob
            
            ensemble_prediction = 1 if ensemble_probability > 50.0 else 0
            predicted_label = self.target_encoder.inverse_transform([ensemble_prediction])[0]
            
            # Create detailed profiles for similar people
            profiles_details = []
            for i, person in enumerate(similar_people):
                profile_detail = {
                    'profile_num': i + 1,
                    'similarity': float(person['similarity_score'] * 100),
                    'age': person.get('age', 'Unknown'),
                    'gender': person.get('sex', 'Unknown'),
                    'education': person.get('education_num', 'Unknown'),
                    'occupation': person.get('occupation_grouped', 'Unknown'),
                    'marital_status': person.get('marital_status', 'Unknown'),
                    'work_hours': person.get('hours_per_week', 'Unknown'),
                    'income': '>50K' if person.get('income', 0) == 1 else '≤50K',
                    'probability': similar_probabilities[i]
                }
                profiles_details.append(profile_detail)
            
            # Create detailed response
            age = user_features['age']
            sex = user_features['sex']
            education_num = user_features['education_num']
            
            education_names = {
                16: 'Doctorate', 15: 'Professional degree', 14: 'Masters', 13: 'Bachelors',
                12: 'Associate degree', 11: 'Associate degree', 10: 'Some college',
                9: 'High school', 8: 'High school', 7: 'Some high school'
            }
            education = education_names.get(education_num, 'High school')
            
            occupation_names = {
                'Prof-specialty': 'Professional/Specialist',
                'Exec-managerial': 'Executive/Manager',
                'Sales': 'Sales',
                'Craft-repair': 'Skilled trades',
                'Machine-op-inspct': 'Machine operator',
                'Other': 'Service worker'
            }
            occupation = occupation_names.get(user_features['occupation_grouped'], 'Professional')
            
            response_details = f"Age: {age}, Gender: {sex}, Education: {education}, Occupation: {occupation}"
            
            if user_features['marital_status'] != 'Never-married':
                marital_display = user_features['marital_status'].replace('-', ' ').title()
                response_details += f", Status: {marital_display}"
            
            if user_features['hours_per_week'] != 40:
                response_details += f", Hours/Week: {user_features['hours_per_week']}"
            
            context_note = f" (Ensemble of {len(similar_people)} similar profiles: {weighted_similar_prob:.1f}% avg, user-specific: {user_probability:.1f}%)"
            
            if predicted_label == '>50K':
                return f"ENSEMBLE PREDICTION: Based on the profile ({response_details}), there is a {ensemble_probability:.1f}% probability of earning over $50K annually. The model predicts income above $50K.{context_note}"
            else:
                return f"ENSEMBLE PREDICTION: Based on the profile ({response_details}), there is a {ensemble_probability:.1f}% probability of earning over $50K annually. The model predicts income of $50K or less.{context_note}"
                
        except Exception as e:
            return f"Error making ensemble prediction: {str(e)}"

class IncomePredictionTool(BaseTool):
    name: str = "income_prediction_tool"
    description: str = """
    Predicts whether a person's annual income exceeds $50K based on demographic information.
    Uses RAG (Retrieval-Augmented Generation) with ensemble prediction from similar profiles.
    
    This tool uses a production XGBoost model with 92.6% AUC performance trained on 39,240
    demographic samples from US Census data. It employs RAG to find similar people in the
    training data and creates ensemble predictions by combining user-specific features
    with weighted averages of similar profiles.
    
    Input: Natural language demographic description (e.g., "35-year-old married software engineer with master's degree")
    Output: Detailed prediction with probability, similar profiles, and analysis
    """
    args_schema: Type[BaseModel] = IncomeInput

    def _run(self, query: str) -> str:
        """Execute the income prediction"""
        try:
            print(f"🤖 Income prediction tool called with: {query}")
            
            # Ensure we have a valid query
            if not query or len(query.strip()) < 3:
                return "Error: Query too short or empty. Please provide demographic information like age, education, occupation, etc."
            
            # Use the IncomePredictor class
            predictor = get_predictor()
            result = predictor.predict_from_query_ensemble(query.strip())
            
            print(f"✅ Tool execution successful")
            return result
            
        except Exception as e:
            error_msg = f"Income prediction tool error: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg

# Initialize global predictor instance
_predictor = None

def get_predictor():
    """Get or initialize the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = IncomePredictor(verbose=True)  # Enable verbose for debugging
    return _predictor

# Create the tool instance - THIS IS THE KEY
income_prediction_tool = IncomePredictionTool()