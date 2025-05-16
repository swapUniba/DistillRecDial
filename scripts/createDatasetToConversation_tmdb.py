import os
import random
import json
import pandas as pd
import numpy as np
from collections import Counter
from datasets import load_dataset, load_from_disk, Dataset, Features, Sequence, Value
from typing import Dict, List, Any, Union
from transformers import AutoTokenizer
import ast
from prompts import ITEMS_FEATURES, SYSTEM_PROMPT


def tqdb_extract(row):
    """Extract TMDB data from a row."""
    usefull_fields = []
    if len(ast.literal_eval(row['casts'])['cast'])>0:
        temp = pd.DataFrame(ast.literal_eval(row['casts'])['cast'])
        temp = temp[temp['known_for_department'].isin(['Acting'])]
        temp = temp.sort_values('popularity', ascending=False).head(10)
        temp = temp.groupby('known_for_department').agg({
            'name': list,
        }).reset_index()
    else:
        temp = []
        for dep in ['Acting']:
            temp.append({'known_for_department': dep, 'name': []})
        temp = pd.DataFrame(temp)
    for _, p in temp.iterrows():
        row[p['known_for_department']] = p['name']
        usefull_fields.append(p['known_for_department'])
    if len(ast.literal_eval(row['casts'])['crew'])>0:
        temp = pd.DataFrame(ast.literal_eval(row['casts'])['crew']).groupby('department').agg({
            'name': list,
        }).reset_index()
    else:
        temp = []
        for dep in ['Directing', 'Sound', 'Production']:
            temp.append({'department': dep, 'name': []})
        temp = pd.DataFrame(temp)
    for _, p in temp.iterrows():
        if p['department'] in ['Directing', 'Sound', 'Production']:
            row[p['department']] = p['name']
            usefull_fields.append(p['department'])
    row['Genres'] = []
    for genre in ast.literal_eval(row['genres']):
        row['Genres'].append(genre['name'])
    usefull_fields.append('Genres')
    row['Keywords'] = []
    for kw in ast.literal_eval(row['keywords'])['keywords']:
        row['Keywords'].append(kw['name'])
    usefull_fields.append('Keywords')
    row['Overview'] = row['overview']
    usefull_fields.append('Overview')
    row['Country'] = ast.literal_eval(row['origin_country'])
    usefull_fields.append('Country')
    row['Collection'] = ''
    if row['belongs_to_collection'] is not None:
        row['Collection'] = ast.literal_eval(row['belongs_to_collection'])['name']
    usefull_fields.append('Collection')
    row['Title'] = row['title']
    usefull_fields.append('Title')
    usefull_fields.append('parent_asin')
    # Remove unnecessary columns
    for col in row.index:
        if col not in usefull_fields:
            del row[col]
    return row

def safe_list_conversion(value: Any) -> List:
    """Safely convert any value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            else:
                return [str(parsed)]
        return [str(value)]
    except (ValueError, SyntaxError):
        return [str(value)]


def safe_string_conversion(value: Any) -> Union[str, None]:
    """Safely convert any value to a string or None."""
    if value is None or value == 'null':
        return ''
    return str(value)


def extract_name_list(items: Any) -> List[str]:
    """Extract names from a list of dictionaries with 'name' key."""
    if not items:
        return []
    
    if isinstance(items, str):
        try:
            items = ast.literal_eval(items)
        except (ValueError, SyntaxError):
            return []
    
    if not isinstance(items, list):
        return []
    
    result = []
    for item in items:
        if isinstance(item, dict) and 'name' in item:
            result.append(item['name'])
    
    return result


def process_tmdb_item(example: Dict) -> Dict:
    """
    Process a TMDB item to extract relevant information.
    This function is designed to be used with dataset.map()
    Ensures consistent types to avoid PyArrow type mixing errors.
    """
    result = {}
    
    # Process casts - handle both string and dict formats
    casts = example.get('casts', {})
    if isinstance(casts, str):
        try:
            casts = ast.literal_eval(casts)
        except (ValueError, SyntaxError):
            casts = {}
    
    if not isinstance(casts, dict):
        casts = {}
    
    # Extract actors (Acting)
    cast_list = casts.get('cast', [])
    if not isinstance(cast_list, list):
        cast_list = []
    
    actors = []
    for item in cast_list:
        if isinstance(item, dict) and item.get('known_for_department') == 'Acting':
            if 'name' in item:
                actors.append(item)
    
    # Sort by popularity
    actors.sort(key=lambda x: float(x.get('popularity', 0)), reverse=True)
    result['Acting'] = [actor['name'] for actor in actors[:10]]
    
    # Process crew by departments
    crew_list = casts.get('crew', [])
    if not isinstance(crew_list, list):
        crew_list = []
    
    for department in ['Directing', 'Sound', 'Production']:
        dept_crew = []
        for person in crew_list:
            if isinstance(person, dict) and person.get('department') == department:
                if 'name' in person:
                    dept_crew.append(person['name'])
        result[department] = dept_crew
    
    # Process genres
    genres = example.get('genres', [])
    if isinstance(genres, str):
        try:
            genres = ast.literal_eval(genres)
        except (ValueError, SyntaxError):
            genres = []
    
    if not isinstance(genres, list):
        genres = []
    
    result['Genres'] = extract_name_list(genres)
    
    # Process keywords
    keywords_data = example.get('keywords', {})
    if isinstance(keywords_data, str):
        try:
            keywords_data = ast.literal_eval(keywords_data)
        except (ValueError, SyntaxError):
            keywords_data = {}
    
    if not isinstance(keywords_data, dict):
        keywords_data = {}
    
    keywords_list = keywords_data.get('keywords', [])
    if not isinstance(keywords_list, list):
        keywords_list = []
    
    result['Keywords'] = extract_name_list(keywords_list)
    
    # Process overview
    result['Overview'] = safe_string_conversion(example.get('overview', ''))
    
    # Process origin country
    country = example.get('origin_country', [])
    result['Country'] = safe_list_conversion(country)
    
    # Process collection
    collection = example.get('belongs_to_collection')
    if collection and collection != 'null':
        if isinstance(collection, str):
            try:
                collection_data = ast.literal_eval(collection)
                if isinstance(collection_data, dict) and 'name' in collection_data:
                    result['Collection'] = collection_data['name']
                else:
                    result['Collection'] = ''
            except (ValueError, SyntaxError):
                result['Collection'] = ''
        elif isinstance(collection, dict) and 'name' in collection:
            result['Collection'] = collection['name']
        else:
            result['Collection'] = ''
    else:
        result['Collection'] = ''
    
    # Copy title and ID
    result['Title'] = safe_string_conversion(example.get('title', ''))
    result['Visual Caption']= safe_string_conversion(example.get('Visual Caption', ''))
    
    # Handle parent_asin
    parent_asin = example.get('parent_asin', '')
    if not parent_asin:
        parent_asin = example.get('asin', '')
    result['parent_asin'] = safe_string_conversion(parent_asin)
    
    # Ensure all lists are actually lists
    for field in ['Acting', 'Directing', 'Sound', 'Production', 'Genres', 'Keywords', 'Country']:
        if field not in result or not isinstance(result[field], list):
            result[field] = []
    
    # Ensure strings are actually strings or None
    for field in ['Overview', 'Title', 'parent_asin', 'Visual Caption']:
        if field not in result:
            result[field] = ''
    
    # Collection can be None
    if 'Collection' not in result:
        result['Collection'] = ''
    
    return result


class AmazonDatasetProcessor:
    def __init__(self, dataset_name, data_path, num_threads=8, k=15, items_in_prompt=5, tmdb_path=None):
        """
        Initialize the Amazon dataset processor.
        
        Args:
            dataset_name (str): Name of the dataset (e.g., 'Movies_and_TV')
            data_path (str): Path to store and load the data
            num_threads (int): Number of threads for parallel processing
            k (int): Minimum number of interactions per user and item
            items_in_prompt (int): Number of items to include in the prompt
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.num_threads = num_threads
        self.k = k
        self.items_in_prompt = items_in_prompt
        self.batch_size = 512
        self.tmdb_path = tmdb_path
        
        self.meta_dataset = None
        self.dataset = None
        self.core_users = None
        self.core_items = None
        self.item_df = None
        self.inter_df = None
        self.group_per_user = None
        self.item_dict = None
        
    def check_already_processed(self):
        """Check if the dataset has already been processed."""
        return f"{self.k}core_rating_only_{self.dataset_name}.hdf5" in os.listdir(self.data_path)
    
    def load_datasets(self):
        """Load the datasets."""
        # Load meta dataset
        self.meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.dataset_name}",
            split="full",
            trust_remote_code=True,
            num_proc=self.num_threads
        )
        
        # Load or process core dataset
        if self.check_already_processed():
            self.dataset = load_from_disk(
                os.path.join(self.data_path, f"{self.k}core_rating_only_{self.dataset_name}.hdf5")
            )
        else:
            self.dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"5core_rating_only_{self.dataset_name}",
                trust_remote_code=True,
                num_proc=self.num_threads
            )['full']
            
        # Print dataset stats
        print(f"Loaded {len(self.dataset)} reviews")
        print(f"Loaded {len(self.meta_dataset)} meta data")
        
    def filter_meta_dataset(self):
        if self.tmdb_path:
            # Load TMDB data
            with open(self.tmdb_path, 'r') as f:
                tmdb_data = json.load(f)
                tmdb_df = pd.DataFrame(tmdb_data)#.set_index('asin')
            print(f"Loaded TMDB data with {len(tmdb_data)} items")

        """Filter the meta dataset to keep only valid items."""
        num_items = len(self.meta_dataset)
        
        # Filter items with title
        self.meta_dataset = self.meta_dataset.filter(
            lambda x: x['title'] is not None, 
            num_proc=self.num_threads
        )
        print(f"Filtered meta data to {len(self.meta_dataset)} items with title")
        print(f"Deleted {num_items - len(self.meta_dataset)} items without title "
              f"(e.g. {(num_items - len(self.meta_dataset)) / num_items:.2%})")
        
        # Filter items with title longer than 2 characters
        num_items = len(self.meta_dataset)
        self.meta_dataset = self.meta_dataset.filter(
            lambda x: len(x['title']) > 2, 
            num_proc=self.num_threads
        )
        print(f"Filtered meta data to {len(self.meta_dataset)} items with title longer than 2 characters")
        print(f"Deleted {num_items - len(self.meta_dataset)} items with title shorter than 2 characters "
              f"(e.g. {(num_items - len(self.meta_dataset)) / num_items:.2%})")
        if self.tmdb_path:
            # Filter items with TMDB data
            items_with_tmdb = set(tmdb_df['asin'].tolist())
            self.meta_dataset = self.meta_dataset.filter(
                lambda x: x['parent_asin'] in items_with_tmdb, 
                num_proc=self.num_threads
            )
            print(f"Filtered meta data to {len(self.meta_dataset)} items with TMDB data")
            print(f"Deleted {num_items - len(self.meta_dataset)} items without TMDB data "
                  f"(e.g. {(num_items - len(self.meta_dataset)) / num_items:.2%})")
        return
    
    def filter_dataset_by_items(self):
        """Filter the dataset to keep only reviews for items in the meta dataset."""
        num_reviews = len(self.dataset)
        items = set(self.meta_dataset['parent_asin'])
        
        self.dataset = self.dataset.filter(
            lambda x: x['parent_asin'] in items, 
            num_proc=self.num_threads
        )
        
        print(f"Filtered dataset to {len(self.dataset)} reviews for {len(items)} items")
        print(f"Deleted {num_reviews - len(self.dataset)} reviews for items without title "
              f"(e.g. {(num_reviews - len(self.dataset)) / num_reviews:.2%})")
    
    def apply_k_core_filtering(self):
        """Apply k-core filtering to ensure each user and item has at least k interactions."""
        while True:
            users_count = Counter(self.dataset['user_id'])
            items_count = Counter(self.dataset['parent_asin'])
            
            if min(users_count.values()) >= self.k and min(items_count.values()) >= self.k:
                print(f"# Items: {len(items_count)}")
                print(f"# User: {len(users_count)}")
                print(f"# Ratings: {len(self.dataset)}")
                break
            
            users_count = Counter({user_id: count for user_id, count in users_count.items() if count >= self.k})
            items_count = Counter({asin: count for asin, count in items_count.items() if count >= self.k})
            
            self.dataset = self.dataset.filter(
                lambda x: x['user_id'] in users_count and x['parent_asin'] in items_count, 
                num_proc=self.num_threads, 
                load_from_cache_file=False
            )
        
        self.core_users = set(self.dataset['user_id'])
        self.core_items = set(self.dataset['parent_asin'])
    
    def load_and_filter_dataframes(self):
        """Load and filter dataframes from JSONL files."""
        if self.tmdb_path:
            # Load TMDB data
            with open(self.tmdb_path, 'r') as f:
                tmdb_data = json.load(f)
            
            print(f"Loaded TMDB data with {len(tmdb_data)} items")
        # Load item and interaction dataframes
        else:
            self.item_df = pd.read_json(
                os.path.join(self.data_path, f'meta_{self.dataset_name}.jsonl'), 
                lines=True
            )
        self.inter_df = pd.read_json(
            os.path.join(self.data_path, f'{self.dataset_name}.jsonl'), 
            lines=True
        )
        
        # Filter for core items and users
        self.inter_df = self.inter_df[self.inter_df['asin'].isin(self.core_items)]
        self.inter_df = self.inter_df[self.inter_df['user_id'].isin(self.core_users)]
        
        # Group interactions by user
        self.group_per_user = self.inter_df.groupby('user_id').agg({
            'rating': list, 
            'asin': list, 
            'title': list, 
            'text': list, 
            'timestamp': list
        }).reset_index()
        
        self.group_per_user['interaction_len'] = self.group_per_user['rating'].apply(lambda x: len(x))
        
        if self.tmdb_path:
            raw_dataset = Dataset.from_list(tmdb_data)
            # Define features for the processed dataset
            features = Features({
                'Acting': Sequence(Value('string')),
                'Directing': Sequence(Value('string')),
                'Sound': Sequence(Value('string')),
                'Production': Sequence(Value('string')),
                'Genres': Sequence(Value('string')),
                'Keywords': Sequence(Value('string')),
                'Overview': Value('string'),
                'Country': Sequence(Value('string')),
                'Collection': Value('string'),
                'Title': Value('string'),
                'Visual Caption': Value('string'),
                'parent_asin': Value('string')
            })
            
            # Process the dataset using map() for efficiency
            processed_dataset = raw_dataset.map(
                process_tmdb_item,
                num_proc=self.num_threads,  # Parallel processing
                remove_columns=raw_dataset.column_names,  # Remove original columns
                features=features,  # Set output features
                desc="Processing TMDB data"  # Progress description
            )
            self.item_df = processed_dataset.to_pandas()
            self.item_dict = self.item_df.set_index('parent_asin').to_dict('index')
        # Filter item dataframe
        else:
            self.item_df = self.item_df.loc[:, [
                'parent_asin', 'title', 'description', 'details', 
                'features', 'price', 'categories'
            ]]
        self.item_df = self.item_df[self.item_df['parent_asin'].isin(self.core_items)]
        self.item_df.fillna("", inplace=True)
        print(f"Filtered item dataframe to {len(self.item_df)} items")
    
    def sample_user_interactions(self):
        """Sample user interactions for prompts."""
        random.seed(42)
        self.group_per_user = self.group_per_user.apply(
            self._get_sample_interactions, 
            n=self.items_in_prompt, 
            axis=1
        )

    def _get_sample_interactions(self, row, n=7):
        """Sample n interactions from a user's history."""
        k = min(row['interaction_len'], n)
        row['all_timestamp'], row['all_rating'], row['all_asin'], row['all_title'], row['all_text'] = zip(*sorted(
            zip(row['timestamp'], row['rating'], row['asin'], row['title'], row['text'])
        ))
        row['timestamp'], row['rating'], row['asin'], row['title'], row['text'] = map(
            lambda lst: lst[-k:-1], 
            [row['timestamp'], row['rating'], row['asin'], row['title'], row['text']]
        )
        row['target_asin'] = row['all_asin'][-1]
        row['target_rating'] = row['all_rating'][-1]
        row['target_title'] = row['all_title'][-1]
        row['target_text'] = row['all_text'][-1]
        return row
    
    def _get_user_prompt(self, row):
        """Generate a prompt for a user based on the scenario."""
        scenario = row['scenario']
        prompt = ""
        if 'history' in ITEMS_FEATURES[scenario] and row.get('asin'):
            history_features = ITEMS_FEATURES[scenario]['history']
            prompt += "The user has interacted with the following items:\n"
            liked_items = ""
            disliked_items = ""
            for i in range(len(row['asin'])):
                temp = ""
                item_info = self.item_dict.get(row['asin'][i], {})
                temp += f"Item: {item_info['Title']}, ID: <id_{row['asin'][i]}>\n"
                if 'Genres' in history_features and len(item_info.get('Genres', [])) > 0:
                    temp += f"Genres: {', '.join(item_info['Genres'])}\n"
                if 'Keywords' in history_features and len(item_info.get('Keywords', [])) > 0:
                    temp += f"Keywords: {', '.join(item_info['Keywords'])}\n"
                if 'Actors' in history_features and len(item_info.get('Actors', [])) > 0:
                    temp += f"Actors: {', '.join(item_info['Actors'])}\n"
                if 'Directors' in history_features and len(item_info.get('Directors', [])) > 0:
                    temp += f"Directors: {', '.join(item_info['Directors'])}\n"
                if 'Collection' in history_features and len(item_info.get('Collection', '')) > 2:
                    temp += f"Collection: {item_info['Collection']}\n"
                if 'Overview' in history_features and len(item_info.get('Overview', '')) > 2:
                    temp += f"Overview: {item_info['Overview']}\n"
                if 'Visual Caption' in history_features and len(item_info.get('Visual Caption', '')) > 2:
                    temp += f"Visual Caption: {item_info['Visual Caption']}\n"
                if "Review" in history_features and len(row['text'][i]) > 2:
                    temp += f"User Review Title: {row['title'][i]}\n"
                    temp += f"User Review: {row['text'][i]}\n"
                if row['rating'][i] > 3:
                    liked_items += f"{temp}\n"
                else:
                    disliked_items += f"{temp}\n"

            if len(liked_items) > 2:
                prompt += f"User liked the following items:\n{liked_items}\n"
            if len(disliked_items) > 2:
                prompt += f"User disliked the following items:\n{disliked_items}\n"    

        if 'target' in ITEMS_FEATURES[scenario]:
            item_info = self.item_dict.get(row['target_asin'], {})
            target_features = ITEMS_FEATURES[scenario]['target']
            prompt += "\nConsider the following one the target item to recommend:\n"
            prompt += f"Item: {self.item_dict.get(row['target_asin'], {}).get('Title', '')}, ID: <id_{row['target_asin']}>\n"
            if 'Genres' in target_features and len(item_info.get('Genres', [])) > 0:
                prompt += f"Genres: {', '.join(item_info['Genres'])}\n"
            if 'Keywords' in target_features and len(item_info.get('Keywords', [])) > 0:
                prompt += f"Keywords: {', '.join(item_info['Keywords'])}\n"
            if 'Actors' in target_features and len(item_info.get('Actors', [])) > 0:
                prompt += f"Actors: {', '.join(item_info['Actors'])}\n"
            if 'Directors' in target_features and len(item_info.get('Directors', [])):
                prompt += f"Directors: {', '.join(item_info['Directors'])}\n"
            if 'Collection' in target_features and len(item_info.get('Collection', '')) > 2:
                prompt += f"Collection: {item_info['Collection']}\n"
            if 'Overview' in target_features and len(item_info.get('Overview', '')) > 2:
                prompt += f"Overview: {item_info['Overview']}\n"
            if 'Visual Caption' in target_features and len(item_info.get('Visual Caption', '')) > 2:
                prompt += f"Visual Caption: {item_info['Visual Caption']}\n"
            if "Review" in target_features and len(row['target_text']) > 2:
                prompt += f"User Review Title: {row['target_title']}\n"
                prompt += f"User Review: {row['target_text']}\n"
        return prompt.strip()
    
    def create_user_prompts(self):
        """Create prompts for each user based on their interactions."""
        random.seed(42)
        self.group_per_user['scenario'] = random.choices(
            list(ITEMS_FEATURES.keys()), 
            k=len(self.group_per_user)
        )
        self.group_per_user.loc[:, 'prompt'] = self.group_per_user.apply(self._get_user_prompt, axis=1)
    
    def format_prompts_with_template(self, tokenizer_path):
        """Format prompts using a tokenizer template."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.group_per_user.loc[:, 'formatted'] = self.group_per_user.apply(
            lambda p: self._get_formatted_prompt(p['prompt'], SYSTEM_PROMPT[p['scenario']], tokenizer), axis=1
        )
    
    def _get_formatted_prompt(self, prompt, system_prompt, tokenizer):
        """Format a prompt using the tokenizer chat template."""
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def save_data(self):
        """Save processed data to files."""
        # Save grouped user data
        self.group_per_user.to_pickle(os.path.join(
            self.data_path, 
            f'{self.dataset_name}_group_per_user_core{self.k}_items{self.items_in_prompt}.pkl'
        ))
        
        # Save item data
        self.item_df.to_pickle(os.path.join(
            self.data_path, 
            f'{self.dataset_name}_item_df_core{self.k}.pkl'
        ))
        
        # Save formatted prompts to JSONL
        with open(os.path.join(
            self.data_path, 
            f'{self.dataset_name}_core{self.k}_items{self.items_in_prompt}_clean_strategy.jsonl'
        ), 'w') as outfile:
            for id_, row in self.group_per_user.set_index('user_id').iterrows():
                json.dump({'user_id': id_, 'scenario': row['scenario'], 'text': row['formatted']}, outfile)
                outfile.write('\n')
    
    def process(self, tokenizer_path):
        """Run the complete processing pipeline."""
        self.load_datasets()
        self.filter_meta_dataset()
        self.filter_dataset_by_items()
        self.apply_k_core_filtering()
        self.load_and_filter_dataframes()
        self.sample_user_interactions()
        self.create_user_prompts()
        self.format_prompts_with_template(tokenizer_path)
        self.save_data()


def main():
    # Constants
    DATA_PATH = './data'
    TMDB_PATH = './tmdb_movies.json'
    DATASET_NAME = 'Movies_and_TV'
    NUM_THREADS = 8
    K_CORE = 12
    ITEMS_IN_PROMPT = 10
    TOKENIZER_PATH = "meta-llama/Llama-3.3-70B-Instruct"
    
    # Process the dataset
    processor = AmazonDatasetProcessor(
        dataset_name=DATASET_NAME,
        data_path=DATA_PATH,
        num_threads=NUM_THREADS,
        k=K_CORE,
        items_in_prompt=ITEMS_IN_PROMPT,
        tmdb_path=TMDB_PATH
    )
    
    processor.process(TOKENIZER_PATH)


if __name__ == "__main__":
    main()