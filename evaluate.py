from unsloth import FastLanguageModel
import torch
import pandas as pd
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage

# Label mapping
LABEL_TO_ID = {
    "before": 0,
    "begins-on": 1,
    "ends-on": 2,
    "contains": 3,
    "overlap": 4,
    "simultaneous": 5,
}

# XML prompt template
PROMPT_TEMPLATE = """ 
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

def load_model() -> Tuple[Any, Any]:
    """Load and configure the model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="final_grpo_model",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
    )
    return model, tokenizer

def generate_output(model: Any, tokenizer: Any, context_sentence: str, 
                   source_word: str, target_word: str) -> List[str]:
    """Generate model output for given input."""
    inputs = tokenizer(
        [PROMPT_TEMPLATE.format(
            context_sentence,
            source_word,
            target_word,
            "",  # output - leave this blank for generation
        )
    ], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=2600,
        use_cache=True,
        temperature=0.6,  # Evaluation parameters from paper
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.batch_decode(outputs)

def process_predictions(df: pd.DataFrame, tokenizer: Any) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Process model predictions and extract labels."""
    predictions = []
    true_labels = []
    wrong_outputs = []
    sequence_lengths = []
    
    label_regexes = [
        r"<Label>(.*?)</Label>",
        r"<Label>(.*?)\n"
    ]
    
    for _, row in df.iterrows():
        text = row['output']
        sequence_lengths.append(len(tokenizer.encode(text)))
        
        # Try both regex patterns
        for regex in label_regexes:
            matches = re.findall(regex, text)
            if len(matches) > 1:
                label_found = False
                for label in LABEL_TO_ID:
                    if label in matches[-1].lower():
                        predictions.append(label)
                        true_labels.append(row['type'])
                        label_found = True
                        break
                    elif label == 'simultaneous':
                        predictions.append('SIMULTANEOUS')
                        true_labels.append(row['type'])
                        label_found = True
                        break
                if label_found:
                    break
        
        if not label_found:
            wrong_outputs.append(text)
    
    return predictions, true_labels, wrong_outputs, sequence_lengths

def evaluate_model(test_file: str, output_file: str) -> None:
    """Evaluate model performance on test data."""
    try:
        # Load model
        model, tokenizer = load_model()
        model = FastLanguageModel.for_inference(model)
        
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Generate predictions
        logger.info("Generating predictions...")
        total_outputs = []
        total_inputs = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            output = generate_output(
                model, tokenizer,
                row['sentence_text'],
                row['event_origin_text'],
                row['event_target_text']
            )[0]
            total_outputs.append(output)
            total_inputs.append(row)
        
        # Save predictions
        test_df['output'] = total_outputs
        test_df['input'] = total_inputs
        test_df.to_csv(output_file, index=False)
        
        # Process predictions
        predictions, true_labels, wrong_outputs, sequence_lengths = process_predictions(test_df, tokenizer)
        
        # Convert labels to numeric format
        pred_true_df = pd.DataFrame({'pred': predictions, 'true': true_labels})
        pred_true_df['true'] = pred_true_df['true'].map(lambda x: {LABEL_TO_ID[x.lower()]}.pop())
        pred_true_df['pred'] = pred_true_df['pred'].map(lambda x: {LABEL_TO_ID[x.lower()]}.pop())
        
        # Calculate and print metrics
        logger.info("Calculating metrics...")
        print("Per-class F1 scores:", f1_score(pred_true_df['true'], pred_true_df['pred'], average=None))
        print("Micro F1 score:", f1_score(pred_true_df['true'], pred_true_df['pred'], average='micro'))
        print("Macro F1 score:", f1_score(pred_true_df['true'], pred_true_df['pred'], average='macro'))
        print("Weighted F1 score:", f1_score(pred_true_df['true'], pred_true_df['pred'], average='weighted'))
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model(
        test_file='datasets_procesados/test/e3c_multilingue_tlink_test.csv',
        output_file='predictions_model_grpo.csv'
    )
