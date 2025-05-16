import json
import os
import re
import difflib
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

class DataProcessor:
    def __init__(self, data_path, domain, k, items, num_threads=8, search_span=50):
        """
        Initialize the data processor with configuration parameters.
        
        Args:
            data_path: Base directory containing data files
            domain: Domain name (e.g., "Movies_and_TV")
            k: Core parameter value
            items: Number of items
            num_threads: Number of parallel processing threads
            search_span: Search span for text matching
        """
        self.data_path = data_path
        self.domain = domain
        self.k = k
        self.items = items
        self.num_threads = num_threads
        self.search_span = search_span
        self.items_name_dict = None
        
    def load_data(self):
        """Load all required data sources and prepare dataframes."""
        # Load response data
        response_file = f'generated_conv_{self.domain}_core{self.k}_items{self.items}_clean_strategy_benchmark.jsonl'
        with open(os.path.join(self.data_path, response_file)) as f:
            data = [json.loads(line) for line in f]
        self.df_response = pd.DataFrame(data)
        self.df_response.drop_duplicates(subset=['prompt'], inplace=True)
        print(f"Loaded {len(self.df_response)} responses")
        
        # Load prompt data
        prompt_file = f'{self.domain}_core{self.k}_items{self.items}_clean_strategy.jsonl'
        self.prompt_df = pd.read_json(os.path.join(self.data_path, prompt_file), lines=True)
        self.prompt_df.drop_duplicates(subset=['text'], inplace=True)
        self.prompt_df.rename(columns={'text': 'prompt'}, inplace=True)
        
        # Merge response with prompts
        self.df_response = self.df_response.merge(self.prompt_df, on='prompt', how='left')
        
        # Load prompt building data
        prompt_build_file = f'{self.domain}_group_per_user_core{self.k}_items{self.items}.pkl'
        self.df_prompt_build = pd.read_pickle(os.path.join(self.data_path, prompt_build_file))
        
        # Load item data and create title dictionary
        item_file = f'{self.domain}_item_df_core{self.k}.pkl'
        self.item_df = pd.read_pickle(os.path.join(self.data_path, item_file))
        self.items_name_dict = self.item_df.set_index('parent_asin').to_dict()['Title']
        
        # Merge response with user and item data
        self.df_response = self.df_response.merge(
            self.df_prompt_build.loc[:, ['user_id', 'asin']], 
            on='user_id', 
            how='left'
        )
        
        return self.df_response
    
    def replace_names_with_id(self, row, n_times=5):
        """
        Replace item titles in response text with their corresponding IDs.
        
        Args:
            row: DataFrame row containing response and item data
            n_times: Number of replacement iterations
        
        Returns:
            Dictionary with cleaned response and metadata
        """
        new_row = {}
        resp = row['response']
        titles = {asin: self.items_name_dict[asin] for asin in row['asin']}
        
        for _ in range(n_times):
            for asin, title in titles.items():
                # Clean title by removing content in parentheses or brackets
                title = re.sub(r"[\(\[].*?[\)\]]", "", title)
                
                # Find matches between response and title
                s = difflib.SequenceMatcher(None, resp, title)
                for block in s.get_matching_blocks():
                    # Only process significant matches (>50% of title)
                    if block.size/len(title) > 0.5:
                        id_tag = f"<id_{asin}>"
                        id_tag_len = len(id_tag)
                        
                        before = resp[:block.a-self.search_span-id_tag_len]
                        resp_span = resp[block.a-self.search_span-id_tag_len:block.a+block.size+self.search_span+id_tag_len]
                        after = resp[block.a+block.size+self.search_span+id_tag_len:]
                        title_to_remove = resp[block.a:block.a+block.size].strip()
                        
                        # If ID tag already exists in span, remove it first
                        if id_tag in resp_span:
                            resp_span = re.sub(r"\(?<id_" + re.escape(asin) + r">\)?", "", resp_span).strip()
                            resp = before + resp_span + after
                            
                        # Replace title with ID tag
                        resp_span = resp_span.replace(title_to_remove, id_tag).strip()
                        resp = before + resp_span + after
        
        new_row['response_clean'] = resp
        new_row['prompt'] = row['prompt']
        new_row['asin'] = row['asin']
        new_row['user_id'] = row['user_id']
        return new_row
    
    def process_responses(self):
        """Process responses to replace item names with ID tags."""
        # Create dataset from dataframe for parallel processing
        self.df_response['response_len'] = self.df_response['response'].apply(len)
        self.df_response['prompt_len'] = self.df_response['prompt'].apply(len)
        self.df_response['len'] = self.df_response['response_len'] + self.df_response['prompt_len']
        #self.df_response = self.df_response[self.df_response['len'] < 2e4]
        print(f"Processing {len(self.df_response)} responses")
        dataset = Dataset.from_pandas(self.df_response)
        
        # Apply name replacement in parallel
        processed_dataset = dataset.map(
            self.replace_names_with_id, 
            num_proc=self.num_threads,
        )
        
        # Convert back to pandas and extract relevant columns
        self.df_processed = processed_dataset.to_pandas()
        data = self.df_processed.loc[:, ['prompt', 'response_clean']].rename(
            columns={'response_clean': 'response'}
        ).to_dict('records')
        
        return data
    
    @staticmethod
    def parse_conversation(text):
        """
        Parse conversation text into structured format.
        
        Args:
            text: Dictionary containing prompt and response
            
        Returns:
            List of conversation turns with role and content
        """
        response = text['response']
        prompt_text = text['prompt']
        blocks = re.split(r'<\|eot_id\|>', prompt_text)
        conversation = []
        
        for block in blocks:
            # Clean up the block and remove special tokens
            cleaned_block = block.strip()
            cleaned_block = re.sub(r'<\|begin_of_text\|>', '', cleaned_block)
            
            # Extract role and content using regex
            role_match = re.match(
                r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*)',
                cleaned_block,
                re.DOTALL
            )
            
            if role_match:
                role = role_match.group(1).strip()
                content = role_match.group(2).strip()
                
                # Replace assistant content with the response
                if role == 'assistant':
                    content = response
                    
                # Clean system content
                if role == 'system':
                    content = content.replace(
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024", 
                        ''
                    ).strip()
                    
                conversation.append({
                    'role': role,
                    'content': content
                })
                
        return conversation
    
    @staticmethod
    def get_conversation_messages(content):
        """
        Extract the last message from formatted conversation content.
        
        Args:
            content: String containing formatted conversation turns
            
        Returns:
            Dictionary with the last conversation turn
        """
        response = content.strip().split('\n')
        current_conversation = []

        for turn in response:
            if turn.startswith('[USR]:'):
                current_conversation.append({'user': turn.replace('[USR]:', '').strip()})
            elif turn.startswith('[AI]:'):
                current_conversation.append({'AI': turn.replace('[AI]:', '').strip()})

        return current_conversation[-1] if current_conversation else {}
    
    def create_tokenized_dataset(self, parsed_conversations):
        """
        Create tokenized dataset from parsed conversations.
        
        Args:
            parsed_conversations: List of parsed conversation structures
            
        Returns:
            List of tokenized conversations
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Apply chat template to each conversation
        tokenized_dataset = [
            tokenizer.apply_chat_template(conversation, tokenize=False) 
            for conversation in parsed_conversations
        ]
        
        return tokenized_dataset
    
    def save_dataset(self, tokenized_dataset):
        """
        Save the processed dataset to JSONL file.
        
        Args:
            tokenized_dataset: List of tokenized conversations
        """
        output_file = f'{self.domain}_core{self.k}_items{self.items}_knowledgeDist_clean_10_per_strategy.jsonl'
        output_path = os.path.join(self.data_path, output_file)
        
        with open(output_path, 'w') as outfile:
            for text in tokenized_dataset:
                json.dump({'text': text}, outfile)
                outfile.write('\n')
        
        print(f"Dataset saved to {output_path}")
    
    def process_pipeline(self):
        """Run the complete data processing pipeline."""
        # Load data
        self.load_data()
        
        # Process responses
        data = self.process_responses()
        
        # Parse conversations
        with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
            parsed_conversations = list(executor.map(self.parse_conversation, data))
        
        # Extract last messages
        last_messages = []
        for conversation in parsed_conversations:
            for turn in conversation:
                if turn['role'] == 'assistant':
                    last_messages.append(self.get_conversation_messages(turn['content']))
        
        # Create tokenized dataset
        tokenized_dataset = self.create_tokenized_dataset(parsed_conversations)
        
        # Save dataset
        self.save_dataset(tokenized_dataset)
        
        return {
            "parsed_conversations": parsed_conversations,
            "last_messages": last_messages,
            "tokenized_dataset": tokenized_dataset
        }


def main():
    # Configuration
    data_path = "./data/"
    domain = "Movies_and_TV"
    k = 12
    items = 10
    
    # Initialize and run processor
    processor = DataProcessor(data_path, domain, k, items)
    processor.process_pipeline()


if __name__ == "__main__":
    main()