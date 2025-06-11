from gradio_client import Client
import os
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime

SPACE_NAME = "anyadecarlo/datasci223"

class PromptComparison:
    def __init__(self):
        os.makedirs("results/part_3", exist_ok=True)
        self.client = Client(SPACE_NAME)

    def get_llm_response(self, prompt: str) -> str:
        try:
            response = self.client.predict(prompt, api_name="/predict")
            return response
        except Exception as e:
            print(f"Gradio client error: {str(e)}")
            return f"[Error: {str(e)}]"

    def create_prompts(self, question: str, examples: List[Tuple[str, str]]) -> Dict[str, str]:
        """Create prompts for different strategies"""
        # Direct Q&A (zero-shot)
        direct_prompt = f"Question: {question}\nAnswer:"
        
        # Chain-of-Thought
        cot_prompt = f"""Question: {question}
Let's think about this step by step:
1. First, let's understand what we're being asked
2. Then, we'll break down the key components
3. Finally, we'll provide a comprehensive answer

Answer:"""
        
        # One-shot
        one_shot_prompt = f"""Question: {examples[0][0]}
Answer: {examples[0][1]}

Question: {question}
Answer:"""
        
        # Few-shot
        few_shot_prompt = "\n\n".join([
            f"Question: {q}\nAnswer: {a}"
            for q, a in examples
        ]) + f"\n\nQuestion: {question}\nAnswer:"
        
        return {
            "direct": direct_prompt,
            "chain_of_thought": cot_prompt,
            "one_shot": one_shot_prompt,
            "few_shot": few_shot_prompt
        }

    def score_response(self, response: str, keywords: List[str]) -> float:
        """Score response based on presence of keywords"""
        response = response.lower()
        found_keywords = sum(1 for keyword in keywords if keyword.lower() in response)
        return found_keywords / len(keywords) if keywords else 0

    def run_comparison(self, questions: List[str], examples: List[Tuple[str, str]], 
                      keywords: Dict[str, List[str]]) -> pd.DataFrame:
        """Run comparison of different prompting strategies"""
        results = []
        
        for question in questions:
            print(f"\nTesting question: {question}")
            prompts = self.create_prompts(question, examples)
            
            for strategy, prompt in prompts.items():
                print(f"\nStrategy: {strategy}")
                print(f"Prompt: {prompt}")
                
                response = self.get_llm_response(prompt)
                print(f"Response: {response}")
                
                # Score the response if we have keywords for this question
                score = self.score_response(response, keywords.get(question, []))
                
                results.append({
                    "question": question,
                    "strategy": strategy,
                    "prompt": prompt,
                    "response": response,
                    "score": score
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"results/part_3/prompt_comparison_{timestamp}.csv", index=False)
        
        return df

def main():
    # Example questions
    questions = [
        "What is the boiling point of nitrogen?",
        "Summarize how the immune system works.",
        "Name 3 symptoms of diabetes.",
        "Translate 'I am very tired' to French.",
        "Give one reason why exercise improves mental health."
    ]
    
    # Example Q&A pairs for few-shot learning
    examples = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the largest planet?", "The largest planet is Jupiter.")
    ]
    
    # Keywords to look for in responses (for scoring)
    keywords = {
        "What is the boiling point of nitrogen?": ["nitrogen", "boiling point", "temperature", "kelvin", "celsius"],
        "Summarize how the immune system works.": ["immune", "defense", "white blood cells", "antibodies", "pathogens"],
        "Name 3 symptoms of diabetes.": ["diabetes", "symptoms", "blood sugar", "thirst", "urination"],
        "Translate 'I am very tired' to French.": ["je suis", "fatigué", "très", "french", "translation"],
        "Give one reason why exercise improves mental health.": ["exercise", "mental health", "endorphins", "stress", "mood"]
    }
    
    # Run comparison
    comparator = PromptComparison()
    results_df = comparator.run_comparison(questions, examples, keywords)
    
    # Print summary
    print("\nResults Summary:")
    print(results_df.groupby("strategy")["score"].mean())

if __name__ == "__main__":
    main() 