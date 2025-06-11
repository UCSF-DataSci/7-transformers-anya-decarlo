import requests
import json
import os
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = "https://anyadecarlo-datasci223.hf.space/api/predict"

class PromptComparison:
    def __init__(self):
        # No model name or API key needed for custom Space
        os.makedirs("results/part_3", exist_ok=True)

    def get_llm_response(self, prompt: str) -> str:
        payload = {"data": [prompt]}
        response = requests.post(API_URL, json=payload)
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            else:
                print("Unexpected response format:", data)
                return "[Error: Unexpected response format]"
        except Exception as e:
            print("Error decoding response:", response.text)
            return f"[Error: {str(e)}]"

    def create_prompts(self, question: str, examples: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Create prompts for different strategies (zero-shot, one-shot, few-shot).
        
        Args:
            question (str): The question to ask
            examples (List[Tuple[str, str]]): List of (question, answer) examples
            
        Returns:
            Dict[str, str]: Dictionary containing prompts for each strategy
        """
        # Zero-shot template
        zero_shot_prompt = f"Question: {question}\nAnswer:"
        
        # One-shot template
        one_shot_prompt = f"""Question: {examples[0][0]}
Answer: {examples[0][1]}

Question: {question}
Answer:"""
        
        # Few-shot template
        few_shot_prompt = "\n\n".join([
            f"Question: {q}\nAnswer: {a}"
            for q, a in examples
        ]) + f"\n\nQuestion: {question}\nAnswer:"
        
        return {
            "zero_shot": zero_shot_prompt,
            "one_shot": one_shot_prompt,
            "few_shot": few_shot_prompt
        }

    def score_response(self, response: str, keywords: List[str]) -> float:
        """
        Score a response based on the presence of expected keywords.
        
        Args:
            response (str): The response to score
            keywords (List[str]): List of keywords to look for
            
        Returns:
            float: Score between 0 and 1
        """
        response = response.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in response)
        return found_keywords / len(keywords) if keywords else 0

    def compare_strategies(self, questions: List[str], examples: List[Tuple[str, str]], 
                         expected_keywords: Dict[str, List[str]]) -> Dict:
        """
        Compare different prompting strategies on a set of questions.
        
        Args:
            questions (List[str]): List of questions to test
            examples (List[Tuple[str, str]]): List of (question, answer) examples
            expected_keywords (Dict[str, List[str]]): Dictionary mapping questions to expected keywords
            
        Returns:
            Dict: Results of the comparison
        """
        results = {
            "raw_responses": {},
            "scores": {},
            "average_scores": {},
            "best_method": None
        }
        
        # Initialize scores dictionary
        scores = {strategy: [] for strategy in ["zero_shot", "one_shot", "few_shot"]}
        
        for question in questions:
            results["raw_responses"][question] = {}
            
            # Create prompts for each strategy
            prompts = self.create_prompts(question, examples)
            
            # Get responses for each strategy
            for strategy, prompt in prompts.items():
                response = self.get_llm_response(prompt)
                results["raw_responses"][question][strategy] = response
                
                # Score the response
                score = self.score_response(response, expected_keywords[question])
                scores[strategy].append(score)
        
        # Calculate average scores
        results["scores"] = scores
        results["average_scores"] = {
            strategy: sum(scores) / len(scores)
            for strategy, scores in scores.items()
        }
        
        # Determine best method
        results["best_method"] = max(
            results["average_scores"].items(),
            key=lambda x: x[1]
        )[0]
        
        return results

    def save_results(self, results: Dict):
        """
        Save results in the required format.
        
        Args:
            results (Dict): Results from compare_strategies
        """
        output_file = "results/part_3/prompting_results.txt"
        
        with open(output_file, "w") as f:
            f.write("# Prompt Engineering Results\n\n")
            
            # Write raw responses
            for question, responses in results["raw_responses"].items():
                f.write(f"## Question: {question}\n\n")
                for strategy, response in responses.items():
                    f.write(f"### {strategy.replace('_', ' ').title()} response:\n")
                    f.write(f"{response}\n\n")
                f.write("-" * 50 + "\n\n")
            
            # Write scores
            f.write("## Scores\n\n")
            
            # Create DataFrame for scores
            df = pd.DataFrame(results["scores"])
            df.index = [q.replace(" ", "_").lower() for q in results["raw_responses"].keys()]
            
            # Add average scores
            df.loc["average"] = [results["average_scores"][s] for s in df.columns]
            
            # Add best method
            df.loc["best_method"] = [results["best_method"] if col == "zero_shot" else "" 
                                   for col in df.columns]
            
            # Write to file
            f.write(df.to_csv())

def main():
    # Initialize the comparison
    pc = PromptComparison()
    
    # Define questions and examples
    questions = [
        "What foods should be avoided by patients with gout?",
        "What medications are commonly prescribed for gout?",
        "How can gout flares be prevented?",
        "Is gout related to diet?",
        "Can gout be cured permanently?"
    ]
    
    examples = [
        ("What are the symptoms of gout?",
         "Gout symptoms include sudden severe pain, swelling, redness, and tenderness in joints, often the big toe."),
        ("How is gout diagnosed?",
         "Gout is diagnosed through physical examination, medical history, blood tests for uric acid levels, and joint fluid analysis to look for urate crystals.")
    ]
    
    expected_keywords = {
        "What foods should be avoided by patients with gout?": 
            ["purine", "red meat", "seafood", "alcohol", "beer", "organ meats"],
        "What medications are commonly prescribed for gout?": 
            ["nsaids", "colchicine", "allopurinol", "febuxostat", "probenecid", "corticosteroids"],
        "How can gout flares be prevented?": 
            ["medication", "diet", "weight", "alcohol", "water", "exercise"],
        "Is gout related to diet?": 
            ["yes", "purine", "food", "alcohol", "seafood", "meat"],
        "Can gout be cured permanently?": 
            ["manage", "treatment", "lifestyle", "medication", "chronic"]
    }
    
    # Compare strategies
    results = pc.compare_strategies(questions, examples, expected_keywords)
    
    # Save results
    pc.save_results(results)

if __name__ == "__main__":
    main() 