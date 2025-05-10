from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
import torch
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_COMPLETION_LENGTH = 1200
MAX_PROMPT_LENGTH = 700
MAX_SEQ_LEN = MAX_COMPLETION_LENGTH + MAX_PROMPT_LENGTH + 100
MAX_REASON_LEN = 72 * 12
NUM_GENERATIONS = 6

# XML prompt template
prompt_classification_grpo_xml = """ 
## Instruction ##
- Extract the start point, end point, and span for the given source and target terms.
- Classify the relation between them using one of: BEFORE, BEGINS-ON, ENDS-ON, CONTAINS, SIMULTANEOUS, OVERLAP.
- Answer with the XML Response Format

## Definitions ##
- A term has a start point (time it begins to exist), an end point (time it ceases to exist), and a span (time between both).

## Inputs ##
- **Context Sentence:** *"{}"* 
- **Source term:** *"{}"*  
- **Target term:** *"{}"*  

## XML Response Format ##
<Source>
    <Meaning>Explain the meaning of the source term</Meaning>
    <Start>Reason about the temporal start point</Start>
    <End>Reason about the temporal end point</End>
    <Span>Reason about the temporal span</Span>
</Source>
<Target>
    <Meaning>Explain the meaning of the target term</Meaning>
    <Start>Reason about the temporal start point</Start>
    <End>Reason about the temporal end point</End>
    <Span>Reason about the temporal span</Span>
</Target>
<Relation>
    <Start_Start>
        <Reason>Reason about the relation of the start and start points</Reason>
        <Answer>before/after/equal/different</Answer>
    </Start_Start>
    <Start_End>
        <Reason>Reason about the relation of the start and end points</Reason>
        <Answer>before/after/equal/different</Answer>
    </Start_End>
    <End_Start>
        <Reason>Reason about the relation of the end and start points</Reason>
        <Answer>before/after/equal/different</Answer>
    </End_Start>
    <End_End>
        <Reason>Reason about the relation of the end and end points</Reason>
        <Answer>before/after/equal/different</Answer>
    </End_End>
    <Label>BEFORE/BEGINS-ON/ENDS-ON/CONTAINS/SIMULTANEOUS/OVERLAP</Label>
</Relation>

<think>\n
    """

class XMLProcessor:
    """Handles XML processing and validation."""
    
    @staticmethod
    def clean_xml_text(text: str) -> str:
        """Clean XML text by removing unnecessary tokens and tags."""
        replacements = {
            '<|endoftext|>': '',
            '</think>\n': '',
            '<think>\n': '',
            '\n</think>': '',
            '\n<think>': '',
            '</think>': '',
            '<think>': ''
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def prepare_xml(output_text: str) -> Tuple[str, List[str], List[str], str, int, int, int]:
        """Process and validate XML output."""
        # Clean input
        output_text = XMLProcessor.clean_xml_text(output_text)
        sentences = output_text.split('\n')
        
        # Initialize flags and containers
        start_xml_flag = -1  # If stays -1, output starts with XML
        end_xml_flag = -1    # If stays -1, output ends with XML
        malformed_xml_flag = -1  # If stays -1, XML is well-formed
        total_reasoning = []
        total_classification_points = []
        total_classification_final = ''

        # Process XML structure
        cleaned_output_text, start_xml_flag, end_xml_flag = XMLProcessor._extract_xml_content(sentences)
        
        # Check for XML duplication
        if XMLProcessor._is_xml_duplicated(cleaned_output_text):
            end_xml_flag = 0

        # Add root element
        cleaned_output_text = f"<root>\n{cleaned_output_text}\n</root>"
        
        # Validate and process XML content
        if XMLProcessor._validate_xml_structure(cleaned_output_text):
            root = ET.fromstring(cleaned_output_text)
            total_reasoning, total_classification_points, total_classification_final = XMLProcessor._extract_xml_data(root)
        else:
            malformed_xml_flag = 0

        return cleaned_output_text, total_reasoning, total_classification_points, total_classification_final, malformed_xml_flag, start_xml_flag, end_xml_flag

    @staticmethod
    def _extract_xml_content(sentences: List[str]) -> Tuple[str, int, int]:
        """Extract XML content from sentences."""
        start_idx = -1
        end_idx = -1
        
        # Find XML start
        for i, sentence in enumerate(sentences):
            if sentence.strip().startswith('<') and sentence.strip().endswith('>'):
                start_idx = i
                break
        
        # Find XML end
        for i, sentence in enumerate(reversed(sentences)):
            if sentence.strip().startswith('<') and sentence.strip().endswith('>'):
                end_idx = len(sentences) - 1 - i
                break
        
        # Extract content
        if start_idx != -1 and end_idx != -1:
            content = '\n'.join(sentences[start_idx:end_idx + 1])
        elif start_idx != -1:
            content = '\n'.join(sentences[start_idx:])
        elif end_idx != -1:
            content = '\n'.join(sentences[:end_idx + 1])
        else:
            content = '\n'.join(sentences)
            
        return content, start_idx, end_idx

    @staticmethod
    def _is_xml_duplicated(text: str) -> bool:
        """Check if XML is duplicated."""
        opening_tags = text.count("<Label>")
        closing_tags = text.count("</Label>")
        return opening_tags > 1 or closing_tags > 1

    @staticmethod
    def _validate_xml_structure(text: str) -> bool:
        """Validate XML structure."""
        try:
            ET.fromstring(text)
            return True
        except ET.ParseError:
            return False

    @staticmethod
    def _extract_xml_data(root: ET.Element) -> Tuple[List[str], List[str], str]:
        """Extract data from XML structure."""
        total_reasoning = []
        total_classification_points = []
        total_classification_final = ''

        try:
            # Extract Source data
            source = root.find('Source')
            if source is not None:
                for tag in ['Meaning', 'Start', 'End', 'Span']:
                    total_reasoning.append(source.find(tag).text if source.find(tag) is not None else "")

            # Extract Target data
            target = root.find('Target')
            if target is not None:
                for tag in ['Meaning', 'Start', 'End', 'Span']:
                    total_reasoning.append(target.find(tag).text if target.find(tag) is not None else "")

            # Extract Relation data
            relation = root.find('Relation')
            if relation is not None:
                for tag in ['Start_Start', 'Start_End', 'End_Start', 'End_End']:
                    element = relation.find(tag)
                    if element is not None:
                        total_reasoning.append(element.find('Reason').text if element.find('Reason') is not None else "")
                        total_classification_points.append(element.find('Answer').text.split()[0] if element.find('Answer') is not None else "")
                
                label = relation.find('Label')
                if label is not None:
                    total_classification_final = label.text.split()[0]

        except Exception as e:
            logger.error(f"Error extracting XML data: {str(e)}")

        return total_reasoning, total_classification_points, total_classification_final

class RewardCalculator:
    """Calculates rewards for model outputs."""
    
    SCORE_WEIGHTS = {
        'xml_format': 0.25,
        'point_classification': 0.2,
        'point_classification_bonus': 0.7,
        'final_classification': 2.0,
        'reasoning_length_factor': 0.001159
    }

    @staticmethod
    def calculate_reward(completions: List[str], **kwargs) -> List[float]:
        """Calculate rewards for model completions."""
        output_texts = completions
        prompts = kwargs["prompts"]
        expected_labels = kwargs["clasificacion_esperada"]
        expected_point_labels = kwargs["clasificacion_point_esperada"]
        tokenizer = kwargs["processing_class"]  # Get tokenizer from kwargs
        
        scores = []
        xml_format_scores = []
        reasoning_length_scores = []
        point_scores = []
        label_scores = []
        
        for output_text, expected_point_label, expected_label in zip(output_texts, expected_point_labels, expected_labels):
            score = 0
            xml_format_score = 0
            reasoning_length_score = 0
            point_score = 0
            label_score = 0
            
            # Process XML
            cleaned_output_text, total_reasoning, total_classification_points, total_classification_final, malformed_xml_flag, start_xml_flag, end_xml_flag = XMLProcessor.prepare_xml(output_text)
            
            if malformed_xml_flag == -1:
                # Calculate XML format score
                xml_format_score = RewardCalculator._calculate_xml_format_score(start_xml_flag, end_xml_flag)
                score += xml_format_score
                
                # Calculate reasoning length score
                reasoning_length_score = RewardCalculator._calculate_reasoning_length_score(total_reasoning, tokenizer)
                score += reasoning_length_score
                
                # Calculate point classification score
                point_score = RewardCalculator._calculate_point_classification_score(
                    total_classification_points, expected_point_label)
                score += point_score
                
                # Calculate final classification score
                label_score = RewardCalculator._calculate_final_classification_score(
                    total_classification_final, expected_label)
                score += label_score
            else:
                score = 0
            
            scores.append(score)
            xml_format_scores.append(xml_format_score)
            reasoning_length_scores.append(reasoning_length_score)
            point_scores.append(point_score)
            label_scores.append(label_score)
        
        # Save results
        RewardCalculator._save_results(prompts, expected_labels, expected_point_labels, 
                                     completions, scores, xml_format_scores, 
                                     reasoning_length_scores, point_scores, label_scores)
        
        return scores

    @staticmethod
    def _calculate_xml_format_score(start_flag: int, end_flag: int) -> float:
        """Calculate score for XML format."""
        score = 0.0
        if start_flag == -1:
            score += RewardCalculator.SCORE_WEIGHTS['xml_format']
        if end_flag == -1:
            score += RewardCalculator.SCORE_WEIGHTS['xml_format']
        return score

    @staticmethod
    def _calculate_reasoning_length_score(reasoning: List[str], tokenizer: Any) -> float:
        """Calculate score for reasoning length."""
        reasoning_str = " ".join(["" if x is None else x for x in reasoning])
        reasoning_length = len(tokenizer.encode(reasoning_str))
        
        if reasoning_length == MAX_REASON_LEN:
            return 1.0
        
        factor = RewardCalculator.SCORE_WEIGHTS['reasoning_length_factor']
        if reasoning_length < MAX_REASON_LEN:
            return reasoning_length * factor
        return 1.0 - (reasoning_length - MAX_REASON_LEN) * factor

    @staticmethod
    def _calculate_point_classification_score(predictions: List[str], expected: List[str]) -> float:
        """Calculate score for point classifications."""
        score = 0.0
        all_correct = True
        
        for pred, exp in zip(predictions, expected):
            if pred.lower() == exp.lower():
                score += RewardCalculator.SCORE_WEIGHTS['point_classification']
            else:
                all_correct = False
                
        if all_correct:
            score += RewardCalculator.SCORE_WEIGHTS['point_classification_bonus']
            
        return score

    @staticmethod
    def _calculate_final_classification_score(prediction: str, expected: str) -> float:
        """Calculate score for final classification."""
        if prediction.lower() == expected.lower():
            return RewardCalculator.SCORE_WEIGHTS['final_classification']
        return 0.0

    @staticmethod
    def _save_results(prompts: List[str], expected_labels: List[str], 
                     expected_point_labels: List[str], completions: List[str],
                     scores: List[float], xml_format_scores: List[float],
                     reasoning_length_scores: List[float], point_scores: List[float],
                     label_scores: List[float]) -> None:
        """Save results to CSV file."""
    df = pd.DataFrame({
       "prompt": [prompts],
            "expected_label": [expected_labels],
            "expected_point_label": [expected_point_labels],
       "completions": [completions],
       "scores": [scores],
            "xml_format_scores": [xml_format_scores],
            "reasoning_length_scores": [reasoning_length_scores],
            "point_scores": [point_scores],
            "label_scores": [label_scores],
        })
        
        path = 'completions_grpo.csv'
    if os.path.exists(path):
       df.to_csv(path, index=False, header=False, mode='a')
        else:
            df.to_csv(path, index=False)

def setup_model() -> Tuple[Any, Any]:
    """Initialize and configure the model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="pretrained_model",
        device_map="balanced",
        fast_inference=True,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.4,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_dataset(dataset_path: str, tokenizer: Any) -> Dataset:
    """Prepare dataset for training."""
    dataset = load_dataset("csv", data_files=dataset_path)
    
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = prompt_classification_grpo_xml.format(
            example['sentence_text'],
            example['event_origin_text'],
            example['event_target_text']
        )
        return {
            'prompt': prompt,
            'expected_label': example["type"],
            'expected_point_label': label2id[example["type"].lower()]
        }
    
    dataset['train'] = dataset['train'].map(format_example)
    return dataset

def get_trainer_config(dataset_size: int) -> GRPOConfig:
    """Get GRPO trainer configuration."""
    return GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        save_steps=round(dataset_size/10),
        max_grad_norm=0.1,
        report_to="none",
        output_dir="output_dir",
        resume_from_checkpoint=True
    )

def main():
    """Main execution function."""
    try:
        # Setup model
        model, tokenizer = setup_model()
        
        # Prepare dataset
        dataset = prepare_dataset('e3c_multilingue_tlink_train.csv', tokenizer)
        
        # Configure and start training
        trainer_config = get_trainer_config(len(dataset['train']))
        trainer = GRPOTrainer(
    model=model,
            args=trainer_config,
            train_dataset=dataset['train'],
            reward_funcs=RewardCalculator.calculate_reward,
    processing_class=tokenizer,
)

        # Train and save model
        trainer.train()
        trainer.save_model("final_grpo_model")
tokenizer.save_pretrained("final_grpo_model")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
