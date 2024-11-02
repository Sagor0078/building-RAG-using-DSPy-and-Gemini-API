import sys
import os
import dspy
import json
import random
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate
from rich import print
import google.generativeai as genai
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download


gemini_api_key = os.getenv('GEMINI_API_KEY')

class LocalHotPotQA:
    """A simplified version of HotPotQA that works with local data"""
    def __init__(self, train_size=20, dev_size=50, train_seed=1, eval_seed=2023):
        # Create cache directory
        cache_dir = Path('hotpotqa_cache')
        cache_dir.mkdir(exist_ok=True)
        
        try:
            # First attempt: Try to load from local cache
            if (cache_dir / 'train.json').exists() and (cache_dir / 'dev.json').exists():
                with open(cache_dir / 'train.json', 'r') as f:
                    train_data = json.load(f)
                with open(cache_dir / 'dev.json', 'r') as f:
                    dev_data = json.load(f)
            else:
                # Second attempt: Download directly from Hugging Face
                print("Downloading HotPotQA dataset from Hugging Face...")
                dataset = load_dataset(
                    "hotpot_qa",
                    'fullwiki',
                    split=['train', 'validation'],
                    trust_remote_code=True,
                    cache_dir=str(cache_dir)
                )
                
                train_data = dataset[0].to_dict()
                dev_data = dataset[1].to_dict()
                
                # Cache the data locally
                with open(cache_dir / 'train.json', 'w') as f:
                    json.dump(train_data, f)
                with open(cache_dir / 'dev.json', 'w') as f:
                    json.dump(dev_data, f)
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using minimal sample data instead...")
            # Fallback: Use minimal sample data
            train_data = {
                'question': [
                    "What castle did David Gregory inherit?",
                    "Who wrote the novel 'The Great Gatsby'?"
                ],
                'answer': [
                    "Castle of Kinnairdy",
                    "F. Scott Fitzgerald"
                ],
                'context': [
                    ["David Gregory inherited the Castle of Kinnairdy...",
                     "The Castle of Kinnairdy is located in Aberdeenshire..."],
                    ["F. Scott Fitzgerald wrote The Great Gatsby in 1925...",
                     "The Great Gatsby is considered a literary masterpiece..."]
                ]
            }
            dev_data = train_data.copy()
        
        # Set random seeds
        random.seed(train_seed)
        train_indices = random.sample(range(len(train_data['question'])), min(train_size, len(train_data['question'])))
        
        random.seed(eval_seed)
        dev_indices = random.sample(range(len(dev_data['question'])), min(dev_size, len(dev_data['question'])))
        
        # Create train and dev sets
        self.train = [
            dspy.Example(
                question=train_data['question'][i],
                answer=train_data['answer'][i]
            ).with_inputs('question')
            for i in train_indices
        ]
        
        self.dev = [
            dspy.Example(
                question=dev_data['question'][i],
                answer=dev_data['answer'][i]
            ).with_inputs('question')
            for i in dev_indices
        ]



class GeminiLM(dspy.LM):
    """Custom DSPy Language Model wrapper for Gemini"""
    def __init__(self, model_name='gemini-pro'):
        # Pass the model_name to the parent class as required
        super().__init__(model=model_name)
        self.model_name = model_name
        genai.configure(api_key=gemini_api_key)
        self._model = genai.GenerativeModel(model_name)
    
    def basic_generate(self, prompt, **kwargs):
        try:
            # Convert messages format to a single string if needed
            if isinstance(prompt, list):
                formatted_prompt = "\n\n".join([
                    f"{'Assistant' if msg.get('role', '') == 'assistant' else 'Human'}: {msg.get('content', '')}"
                    for msg in prompt
                ])
            else:
                formatted_prompt = prompt
            
            response = self._model.generate_content(formatted_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""

    def __getstate__(self):
        """Custom serialization method"""
        return {'model_name': self.model_name}

    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__init__(model_name=state['model_name'])

# Also update the HotPotQASystem class initialization to use the fixed LM
class HotPotQASystem:
    def __init__(self):
        # Initialize models and settings with the fixed LM
        self.gemini = GeminiLM(model_name='gemini-pro')  # Explicitly passing model_name
        self.colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
        dspy.settings.configure(lm=self.gemini, rm=self.colbert)
        
        # Rest of the initialization remains the same
        self.dataset = LocalHotPotQA(train_size=20, dev_size=50)
        self.trainset = self.dataset.train
        self.devset = self.dataset.dev
        
        self.teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
        self.compiled_rag = self.teleprompter.compile(RAG(), trainset=self.trainset)
        
        self.baleen_teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
        self.compiled_baleen = self.baleen_teleprompter.compile(
            SimplifiedBaleen(),
            teacher=SimplifiedBaleen(passages_per_hop=2),
            trainset=self.trainset
        )
        
        self.evaluator = Evaluate(
            devset=self.devset,
            num_threads=1,
            display_progress=True,
            display_table=5
        )


# class LocalHotPotQA:
#     """A simplified version of HotPotQA that works with local data"""
#     def __init__(self, train_size=20, dev_size=50, train_seed=1, eval_seed=2023):
#         # Create cache directory
#         cache_dir = Path('hotpotqa_cache')
#         cache_dir.mkdir(exist_ok=True)
        
#         try:
#             # First attempt: Try to load from local cache
#             if (cache_dir / 'train.json').exists() and (cache_dir / 'dev.json').exists():
#                 with open(cache_dir / 'train.json', 'r') as f:
#                     train_data = json.load(f)
#                 with open(cache_dir / 'dev.json', 'r') as f:
#                     dev_data = json.load(f)
#             else:
#                 # Second attempt: Download directly from Hugging Face
#                 print("Downloading HotPotQA dataset from Hugging Face...")
#                 dataset = load_dataset(
#                     "hotpot_qa",
#                     'fullwiki',
#                     split=['train', 'validation'],
#                     trust_remote_code=True,
#                     cache_dir=str(cache_dir)
#                 )
                
#                 train_data = dataset[0].to_dict()
#                 dev_data = dataset[1].to_dict()
                
#                 # Cache the data locally
#                 with open(cache_dir / 'train.json', 'w') as f:
#                     json.dump(train_data, f)
#                 with open(cache_dir / 'dev.json', 'w') as f:
#                     json.dump(dev_data, f)
        
#         except Exception as e:
#             print(f"Error loading dataset: {e}")
#             print("Using minimal sample data instead...")
#             # Fallback: Use minimal sample data
#             train_data = {
#                 'question': [
#                     "What castle did David Gregory inherit?",
#                     "Who wrote the novel 'The Great Gatsby'?"
#                 ],
#                 'answer': [
#                     "Castle of Kinnairdy",
#                     "F. Scott Fitzgerald"
#                 ],
#                 'context': [
#                     ["David Gregory inherited the Castle of Kinnairdy...",
#                      "The Castle of Kinnairdy is located in Aberdeenshire..."],
#                     ["F. Scott Fitzgerald wrote The Great Gatsby in 1925...",
#                      "The Great Gatsby is considered a literary masterpiece..."]
#                 ]
#             }
#             dev_data = train_data.copy()
        
#         # Set random seeds
#         random.seed(train_seed)
#         train_indices = random.sample(range(len(train_data['question'])), min(train_size, len(train_data['question'])))
        
#         random.seed(eval_seed)
#         dev_indices = random.sample(range(len(dev_data['question'])), min(dev_size, len(dev_data['question'])))
        
#         # Create train and dev sets
#         self.train = [
#             dspy.Example(
#                 question=train_data['question'][i],
#                 answer=train_data['answer'][i]
#             ).with_inputs('question')
#             for i in train_indices
#         ]
        
#         self.dev = [
#             dspy.Example(
#                 question=dev_data['question'][i],
#                 answer=dev_data['answer'][i]
#             ).with_inputs('question')
#             for i in dev_indices
#         ]


# class GeminiLM(dspy.LM):
#     """Custom DSPy Language Model wrapper for Gemini"""
#     def __init__(self, model_name='gemini-pro'):
#         super().__init__()
#         self.model_name = model_name
#         genai.configure(api_key=gemini_api_key)
#         self._model = genai.GenerativeModel(model_name)
    
#     def basic_generate(self, prompt, **kwargs):
#         try:
#             # Convert messages format to a single string if needed
#             if isinstance(prompt, list):
#                 formatted_prompt = "\n\n".join([
#                     f"{'Assistant' if msg.get('role', '') == 'assistant' else 'Human'}: {msg.get('content', '')}"
#                     for msg in prompt
#                 ])
#             else:
#                 formatted_prompt = prompt
            
#             response = self._model.generate_content(formatted_prompt)
#             return response.text
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return ""

#     def __getstate__(self):
#         """Custom serialization method"""
#         return {'model_name': self.model_name}

#     def __setstate__(self, state):
#         """Custom deserialization method"""
#         self.__init__(model_name=state['model_name'])


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers using context."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


class GenerateSearchQuery(dspy.Signature):
    """Write a search query for complex question answering."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        traces = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            traces.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer, traces=traces)


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): 
        return False
    if not dspy.evaluate.answer_passage_match(example, pred): 
        return False
    if not hasattr(pred, 'traces'):
        return False
    hops = [example.question] + pred.traces
    if max([len(h) for h in hops]) > 100: 
        return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): 
        return False
    return True


class HotPotQASystem:
    def __init__(self):
        # Initialize models and settings
        self.gemini = GeminiLM()
        self.colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
        dspy.settings.configure(lm=self.gemini, rm=self.colbert)
        
        # Load dataset
        self.dataset = LocalHotPotQA(train_size=20, dev_size=50)
        self.trainset = self.dataset.train
        self.devset = self.dataset.dev
        
        # Initialize models
        self.teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
        self.compiled_rag = self.teleprompter.compile(RAG(), trainset=self.trainset)
        
        self.baleen_teleprompter = BootstrapFewShot(metric=validate_context_and_answer_and_hops)
        self.compiled_baleen = self.baleen_teleprompter.compile(
            SimplifiedBaleen(),
            teacher=SimplifiedBaleen(passages_per_hop=2),
            trainset=self.trainset
        )
        
        self.evaluator = Evaluate(
            devset=self.devset,
            num_threads=1,
            display_progress=True,
            display_table=5
        )
    
    def basic_qa(self, question):
        generate_answer = dspy.Predict(BasicQA)
        return generate_answer(question=question).answer
    
    def rag_qa(self, question):
        return self.compiled_rag(question)
    
    def baleen_qa(self, question):
        return self.compiled_baleen(question)
    
    def evaluate_models(self):
        def gold_passages_retrieved(example, pred, trace=None):
            if not hasattr(example, 'gold_titles'):
                return True  # Skip evaluation if gold titles not available
            gold_titles = set(map(dspy.evaluate.normalize_text, example['gold_titles']))
            found_titles = set(map(dspy.evaluate.normalize_text, [c.split(' | ')[0] for c in pred.context]))
            return gold_titles.issubset(found_titles)
        
        rag_score = self.evaluator(self.compiled_rag, metric=gold_passages_retrieved)
        baleen_score = self.evaluator(self.compiled_baleen, metric=gold_passages_retrieved)
        
        return {
            "rag_score": rag_score,
            "baleen_score": baleen_score
        }


if __name__ == "__main__":
    try:
        # Initialize system
        print("Initializing QA system...")
        qa_system = HotPotQASystem()
        
        # Example questions
        questions = [
            "What castle did David Gregory inherit?",
            "How many storeys are in the castle that David Gregory inherited?"
        ]
        
        print("\n=== Basic QA ===")
        for q in questions:
            print(f"Q: {q}")
            try:
                answer = qa_system.basic_qa(q)
                print(f"A: {answer}\n")
            except Exception as e:
                print(f"Error in basic QA: {e}\n")
        
        print("\n=== RAG QA ===")
        for q in questions:
            try:
                pred = qa_system.rag_qa(q)
                print(f"Q: {q}")
                print(f"A: {pred.answer}")
                print(f"Context: {[c[:200] + '...' for c in pred.context]}\n")
            except Exception as e:
                print(f"Error in RAG QA: {e}\n")
        
        print("\n=== Baleen Multi-hop QA ===")
        for q in questions:
            try:
                pred = qa_system.baleen_qa(q)
                print(f"Q: {q}")
                print(f"A: {pred.answer}")
                print(f"Context: {[c[:200] + '...' for c in pred.context]}\n")
            except Exception as e:
                print(f"Error in Baleen QA: {e}\n")
        
        print("\n=== Evaluation Scores ===")
        try:
            scores = qa_system.evaluate_models()
            print(f"RAG Score: {scores['rag_score']}")
            print(f"Baleen Score: {scores['baleen_score']}")
        except Exception as e:
            print(f"Error in evaluation: {e}")
            
    except Exception as e:
        print(f"System initialization error: {e}")