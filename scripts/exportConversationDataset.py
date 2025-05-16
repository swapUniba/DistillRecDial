import json
import os
import re
import random
from typing import Dict, List, Tuple, Set, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str
    content: str


class AmazonReviewProcessor:
    """
    Processes Amazon review data for conversational recommendation systems.
    
    This class handles loading, processing, and formatting conversational data
    from Amazon reviews, with special handling for item identifiers and their
    replacement with actual product titles.
    """
    
    def __init__(
        self, 
        data_path: str,
        file_name: str,
        dataset_name: str,
        special_token: str,
        num_threads: int = 8,
        k: int = 20,
        items_in_prompt: int = 5,
        max_turns: int = 15,
        tokenizer_path: str = None,
        train_split: float = 0.8,
        valid_split: float = 0.1,
        test_split: float = 0.1,
        stratify: bool = False
    ):
        """
        Initialize the Amazon Review Processor.
        
        Args:
            data_path: Path to the data directory
            dataset_name: Name of the dataset (e.g., 'Movies_and_TV')
            special_token: Special token for marking recommended items
            num_threads: Number of threads for parallel processing
            k: Core parameter value
            items_in_prompt: Number of items in each prompt
            max_turns: Maximum number of conversation turns to keep
            tokenizer_path: Path to the tokenizer model
            train_split: Percentage of data for training set (default: 0.8)
            valid_split: Percentage of data for validation set (default: 0.1)
            test_split: Percentage of data for test set (default: 0.1)
        """
        self.data_path = data_path
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.domain = dataset_name
        self.special_token = special_token
        self.num_threads = num_threads
        self.k = k
        self.items_in_prompt = items_in_prompt
        self.max_turns = max_turns
        self.tokenizer_path = tokenizer_path or "meta-llama/Llama-3.1-8B-Instruct"
        self.stratify = stratify
        
        # Data split percentages
        total = train_split + valid_split + test_split
        if not abs(total - 1.0) < 1e-10:
            raise ValueError(f"Split percentages must sum to 1.0, got {total}")
        
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        
        # Role mapping
        self.role_dict = {'AI': 'assistant', 'user': 'user'}
        
        # Will be populated during processing
        self.item_dict = {}
        self.conversations = []
        self.processed_conversations = []
        self.tokenizer = None

    def load_data(self) -> List[Dict]:
        """
        Load the expected generation data from JSONL file.
        
        Returns:
            List of dictionaries containing response data
        """
        file_path = os.path.join(self.data_path, f'{self.file_name}')
        with open(file_path) as f:
            data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(data)} items from {self.file_name}")
        return data
    
    def load_metadata(self) -> Dict[str, str]:
        """
        Load metadata from the McAuley-Lab/Amazon-Reviews-2023 dataset.
        
        Returns:
            Dictionary mapping item IDs to their titles
        """
        
        item_df = pd.read_pickle(os.path.join(self.data_path, f"{self.dataset_name}_item_df_core{self.k}.pkl"))
        raw_item_dict = item_df.set_index('parent_asin').to_dict()['Title']


        # Clean titles and format with ID tags
        item_dict = {
            f"<id_{key}>": re.sub(r"[\(\[].*?[\)\]]", "", str(value))
            for key, value in raw_item_dict.items()
        }
        
        print(f"Loaded metadata for {len(item_dict)} items")
        return item_dict
    
    def parse_conversations(self, data: List[Dict]) -> List[List[Dict]]:
        """
        Parse response data into structured conversations.
        
        Args:
            data: List of dictionaries with response data
            
        Returns:
            List of conversations, where each conversation is a list of turns
        """
        conversations = []
        
        for file_line in data:
            response = file_line['response']
            current_conversation = []
            
            # Split by the tokens [USR] and [AI]
            turns = re.split(r'(\[USR\]:?|\[AI\]:?)', response)
            turns = [turn.strip() for turn in turns if turn.strip()]
            
            i = 0
            while i < len(turns):
                if turns[i] in ['[USR]', '[USR]:']:
                    if i + 1 < len(turns):
                        current_conversation.append({'user': turns[i+1].strip()})
                        i += 2
                    else:
                        i += 1
                elif turns[i] in ['[AI]', '[AI]:']:
                    if i + 1 < len(turns):
                        current_conversation.append({'AI': turns[i+1].strip()})
                        i += 2
                    else:
                        i += 1
                else:
                    # Handle case where there's content without a preceding token
                    i += 1
            
            if current_conversation:
                target = file_line['target_asin']
                conversations.append({'conversation': current_conversation, 'scenario': file_line['scenario'], 'target': f'<id_{target}>'})
        
        print(f"Parsed {len(conversations)} conversations")
        return conversations
    
    def replace_ids_with_names(self, conversation: List[Dict]) -> Tuple[List[Dict], List[str], int]:
        """
        Replace item IDs with their actual names in a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Tuple containing:
                - Updated conversation with IDs replaced by names
                - List of targeted item IDs
                - Count of items that weren't found in the metadata
        """
        conversation_ = conversation.copy()
        pattern_id = re.compile(r'<id_(.*?)>')
        already_mentioned = set()
        new_conversation = []
        items_target = []
        mentioned_not_targeted = set()
        not_found = 0
        conversation = conversation_['conversation']
        scenario = conversation_['scenario']

        for turn_dict in conversation:
            role, turn = tuple(turn_dict.items())[0]
            matched_items = pattern_id.findall(turn)
            
            for match in matched_items:
                item_id = f'<id_{match}>'
                
                if item_id not in self.item_dict:
                    not_found += 1
                    continue
                
                mentioned_not_targeted.add(item_id)
                if match not in already_mentioned:
                    already_mentioned.add(match)
                    if role == 'AI':
                        # Special handling for assistant responses
                        replacement = f"{self.special_token} {self.item_dict.get(item_id, item_id)}"
                        turn = turn.replace(item_id, replacement).strip()
                        items_target.append(item_id)
                    else:
                        # Standard replacement for user messages
                        turn = turn.replace(item_id, self.item_dict.get(item_id, item_id)).strip()
                else:
                    # Item already mentioned before
                    turn = turn.replace(item_id, self.item_dict.get(item_id, item_id)).strip()
                    
            turn_dict[role] = turn.strip()
            new_conversation.append(turn_dict)
        return new_conversation, items_target, list(mentioned_not_targeted.difference(set(items_target))), not_found, scenario
    
    def replace_ids_with_names_optimized(self, conversation_data: Dict) -> Tuple[List[Dict], List[str], List[str], int, Any]:
        """
        Optimized version of replace_ids_with_names.
        """
        conversation = conversation_data['conversation']
        scenario = conversation_data['scenario']
        target = [conversation_data['target']]
        pattern_id = re.compile(r'<id_(.*?)>')
        item_dict = self.item_dict
        special_token = self.special_token

        new_conversation = []
        items_target = set()
        mentioned_not_targeted = set()
        not_found = 0

        for turn_dict in conversation:
            role, turn = tuple(turn_dict.items())[0]
            matched_ids = pattern_id.findall(turn)
            replacements = {}

            for match in matched_ids:
                item_id_full = f'<id_{match}>'
                item_name = item_dict.get(item_id_full)

                if item_name is None:
                    not_found += 1
                    continue

                mentioned_not_targeted.add(item_id_full)

                if item_id_full not in replacements:  # Avoid redundant replacements
                    if role == 'AI':
                        replacement = f"{special_token} {item_name}"
                        items_target.add(item_id_full)
                    else:
                        replacement = item_name
                    replacements[item_id_full] = replacement

            # Perform all replacements in the current turn at once
            for old, new in replacements.items():
                turn = turn.replace(old, new).strip()

            turn_dict[role] = turn
            new_conversation.append(turn_dict)

        return (
            new_conversation,
            target,
            list(mentioned_not_targeted),
            not_found,
            scenario,
        )


    def process_conversations(self) -> List[Tuple[List[Dict], List[str]]]:
        """
        Process all conversations by replacing IDs with names.
        
        Uses parallel processing to improve performance.
        
        Returns:
            List of tuples containing processed conversations and their target items
        """
        # Process conversations in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(tqdm(
                executor.map(self.replace_ids_with_names_optimized, self.conversations),
                total=len(self.conversations),
                desc="Processing conversations"
            ))
        
        # Filter out conversations with missing items
        valid_conversations = [(x[0], x[1], x[2], x[4]) for x in results if len(x[1]) != 0]
        
        print(f"Conversations with missing target: {len(self.conversations) - len(valid_conversations)}")
        print(f"Total conversations: {len(self.conversations)}")
        print(f"Valid conversations: {len(valid_conversations)}")
        
        # Filter out conversations that are too long
        #filtered_conversations = [x for x in valid_conversations if len(x[0]) < self.max_turns]
        filtered_conversations = valid_conversations

        #print(f"Conversations with more than {self.max_turns} turns: {len(valid_conversations) - len(filtered_conversations)}")
        print(f"Total valid conversations: {len(valid_conversations)}")
        print(f"Filtered conversations: {len(filtered_conversations)}")
        
        return filtered_conversations
    
    def format_conversations(self, processed_conversations: List[Tuple[List[Dict], List[str]]]) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Format conversations for tokenization.
        
        Args:
            processed_conversations: List of processed conversations with target items
            
        Returns:
            Tuple containing:
                - List of formatted conversations
                - List of target dictionaries
        """
        conversation_format = [
            [{"role": self.role_dict[tuple(turn.keys())[0]], "content": turn[tuple(turn.keys())[0]]} 
             for turn in conversation[0]] + [{'target': conversation[1]}] + [{'named_items': conversation[2]}] + [{'scenario': conversation[3]}] 
            for conversation in processed_conversations
        ]
        
        system_prompt = (
            f"You are a friendly and knowledgeable conversational recommender system "
            f"designed to assist users in discovering TV shows, movies, or other content "
            f"tailored to their preferences. Your goal is to provide personalized "
            f"recommendations by understanding the user's tastes, preferences, and viewing "
            f"history. You should engage in natural, conversational dialogue, ask clarifying "
            f"questions when needed, and offer thoughtful suggestions that align with the "
            f"user's interests."
        )
        
        new_conversations = []
        targets = []
        named_items = []
        scenarios = []
        
        for line in conversation_format:
            # Add system prompt to the beginning
            new_line = [{"role": "system", "content": system_prompt}] + line
            new_conversations.append(new_line[:-3])  # Conversation without target
            targets.append(line[-3])  # Target
            named_items.append(line[-2]) # Named items
            scenarios.append(line[-1])

        return new_conversations, targets, named_items, scenarios
    
    def tokenize_conversations(self, conversations: List[List[Dict]]) -> List[str]:
        """
        Tokenize conversations using the model's chat template.
        
        Args:
            conversations: List of formatted conversations
            
        Returns:
            List of tokenized conversations as strings
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply chat template to each conversation
        tokenized = [
            self.tokenizer.apply_chat_template(conv, tokenize=False) 
            for conv in conversations
        ]
        
        return tokenized
    
    def stratified_split(self, dataset, test_size=0.15, val_size=0.15, random_seed=None):
        """
        Perform stratified train, test, validation split based on the 'scenario' attribute.
        
        Args:
            dataset: List of dictionaries containing the instances
            test_size: Proportion of data for test set (default: 0.15)
            val_size: Proportion of data for validation set (default: 0.15)
            random_seed: Seed for reproducibility (default: None)
        
        Returns:
            train_set, test_set, val_set: Three lists containing the split data
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Group instances by scenario
        scenario_groups = defaultdict(list)
        for instance in dataset:
            scenario = instance['scenario']
            scenario_groups[scenario].append(instance)
        
        train_set = []
        test_set = []
        val_set = []
        
        # For each scenario group, perform the split
        for scenario, instances in scenario_groups.items():
            n = len(instances)
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            n_train = n - n_test - n_val
            
            # Shuffle the instances
            random.shuffle(instances)
            
            # Split the group
            train_set.extend(instances[:n_train])
            val_set.extend(instances[n_train:n_train + n_val])
            test_set.extend(instances[n_train + n_val:])
        
        # Shuffle the final sets (optional)
        random.shuffle(train_set)
        random.shuffle(test_set)
        random.shuffle(val_set)
        
        return train_set, test_set, val_set


    def split_dataset(self, tokenized_conversations: List[str], targets: List[Dict], named_items: List[Dict], scenarios: List[Dict]) -> Tuple[List, List, List]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            tokenized_conversations: List of tokenized conversations
            targets: List of target dictionaries
            
        Returns:
            Tuple containing train, validation, and test datasets
        """

        # Create dataset items with both text and target
        dataset = []
        for line, target, named_item, scenarios in zip(tokenized_conversations, targets, named_items, scenarios):
            if self.special_token in line:
                dataset.append({'text': line, **target, **named_item, **scenarios})
        
        if self.stratify:
            train_dataset, test_dataset, valid_dataset = self.stratified_split(dataset, test_size=self.test_split, val_size=self.valid_split, random_seed=42)
        else:
            # Shuffle the dataset
            random.seed(42)  # For reproducibility
            random.shuffle(dataset)
            
            # Calculate split indices
            total_size = len(dataset)
            train_size = int(total_size * self.train_split)
            valid_size = int(total_size * self.valid_split)
            
            # Split the dataset
            train_dataset = dataset[:train_size]
            valid_dataset = dataset[train_size:train_size + valid_size]
            test_dataset = dataset[train_size + valid_size:]
            
            print(f"Dataset split: Train {len(train_dataset)}, Validation {len(valid_dataset)}, Test {len(test_dataset)}")
        
        return train_dataset, valid_dataset, test_dataset
    
    def save_split_datasets(self, train_dataset: List, valid_dataset: List, test_dataset: List) -> None:
        """
        Save train, validation, and test datasets to separate files.
        
        Args:
            train_dataset: List of training data samples
            valid_dataset: List of validation data samples
            test_dataset: List of test data samples
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.data_path, 'distillrecdial')
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean special token for filename
        clean_token = self.special_token.replace("|", "").replace("<", "").replace(">", "")
        
        # Create base filename
        base_filename = f'{self.domain}_llama3.1_{clean_token}'
        
        # Save train dataset
        train_file = os.path.join(output_dir, f'{base_filename}_train.jsonl')
        with open(train_file, 'w') as f:
            for item in train_dataset:
                f.write(json.dumps(item) + '\n')
        
        # Save validation dataset
        valid_file = os.path.join(output_dir, f'{base_filename}_valid.jsonl')
        with open(valid_file, 'w') as f:
            for item in valid_dataset:
                f.write(json.dumps(item) + '\n')
        
        # Save test dataset
        test_file = os.path.join(output_dir, f'{base_filename}_test.jsonl')
        with open(test_file, 'w') as f:
            for item in test_dataset:
                f.write(json.dumps(item) + '\n')
        
        print(f"Train dataset saved to {train_file} ({len(train_dataset)} samples)")
        print(f"Validation dataset saved to {valid_file} ({len(valid_dataset)} samples)")
        print(f"Test dataset saved to {test_file} ({len(test_dataset)} samples)")
    
    def check_tokenization(self, conversation_format: List[List[Dict]]) -> int:
        """
        Check for instances without the target token after tokenization.
        
        Args:
            conversation_format: List of formatted conversations
            
        Returns:
            Number of conversations without the target token
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find the token ID for the special token
        special_token_id = self.tokenizer.encode(self.special_token)[0]
        
        # Check if each tokenized conversation contains the special token
        tokenized = [
            special_token_id in self.tokenizer.apply_chat_template(line, tokenize=True)
            for line in conversation_format
        ]
        
        missing_targets = len(conversation_format) - sum(tokenized)
        print(f"Instances without target: {missing_targets}")
        
        return missing_targets
    
    def process(self) -> None:
        """
        Run the complete data processing pipeline.
        """
        # Load data
        data = self.load_data()
        
        # Load metadata
        self.item_dict = self.load_metadata()
        
        # Parse conversations
        self.conversations = self.parse_conversations(data)
        
        # Process conversations
        processed_conversations = self.process_conversations()
        
        # Format conversations
        formatted_conversations, targets, named_items, scenarios = self.format_conversations(processed_conversations)
        
        # Tokenize conversations
        tokenized_conversations = self.tokenize_conversations(formatted_conversations)
        
        # Split the dataset
        train_dataset, valid_dataset, test_dataset = self.split_dataset(tokenized_conversations, targets, named_items, scenarios)
        
        # Save the split datasets
        self.save_split_datasets(train_dataset, valid_dataset, test_dataset)
        
        # Check tokenization (for quality assurance)
        self.check_tokenization(formatted_conversations)


def main():
    # Configuration
    data_path = './data'
    dataset_name = 'Movies_and_TV'
    file_name = 'generated_conv_Movies_and_TV_core12_items10_8B_LORA_strategy_pipe.jsonl'
    special_token = "<|reserved_special_token_10|>"
    num_threads = 16
    k = 12
    items_in_prompt = 10
    stratify = True
    
    # Data split percentages
    TRAIN_SPLIT = 0.8
    VALID_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Initialize and run processor
    processor = AmazonReviewProcessor(
        data_path=data_path,
        file_name=file_name,
        dataset_name=dataset_name,
        special_token=special_token,
        num_threads=num_threads,
        k=k,
        items_in_prompt=items_in_prompt,
        train_split=TRAIN_SPLIT,
        valid_split=VALID_SPLIT,
        test_split=TEST_SPLIT, 
        stratify=stratify
    )
    
    processor.process()


if __name__ == "__main__":
    main()