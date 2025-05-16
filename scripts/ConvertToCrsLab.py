import os
import re
import json
import pandas as pd
import swifter
import itertools
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer
from EntityMentionDetector import EntityMentionDetector


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_gpt2(metadata:Dict[str, str], results:List[List[Dict]], output_path:str, split:str):
    # Format for CRSLab
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    crs_res = list(map(lambda x: format_CRSLab(x, metadata, tokenizer), results))
    crs_res = [{"conv_id": str(i), "conv": conv} for i, conv in enumerate(crs_res)]
    
    # Save split-specific output
    with open(os.path.join(output_path, f'{split}_data.json'), 'w') as f:
        json.dump(crs_res, f, ensure_ascii=False, indent=4)
    # Save vocab of tokenizer
    with open(os.path.join(output_path, f'token2id.json'), 'w') as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False)

def process_bert(metadata:Dict[str, str], results:List[List[Dict]], output_path:str, split:str):
    # Format for BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    crs_res = list(map(lambda x: format_CRSLab(x, metadata, tokenizer), results))
    crs_res = [{"conv_id": str(i), "conv": conv} for i, conv in enumerate(crs_res)]
    
    # Save split-specific output
    with open(os.path.join(output_path, f'{split}_data.json'), 'w') as f:
        json.dump(crs_res, f, ensure_ascii=False, indent=4)
    # Save vocab of tokenizer
    with open(os.path.join(output_path, f'token2id.json'), 'w') as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False)

def process_nltk(metadata:Dict[str, str], results:List[List[Dict]], output_path:str, split:str):
    raise NotImplementedError("NLTK tokenizer processing is not implemented yet.")


def process_data(data_path: str, output_path: str, amazon_dataset_name: str, splits: List[str], config_tokenizer: str):
    """
    Process the dataset for multiple splits (train, test, valid)
    
    Args:
        data_path: Path to input data
        output_path: Path for output files
        amazon_dataset_name: Name of the Amazon dataset
        splits: List of splits to process (e.g. ['train', 'test', 'valid'])
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load common data that's the same across all splits
    groupd = pd.read_pickle(os.path.join(data_path, f'{amazon_dataset_name}_group_per_user_core12_items10.pkl'))
    items = groupd.swifter.apply(get_item_sampled, axis=1)
    
    # Load metadata
    metadata = load_metadata(data_path, amazon_dataset_name)
    
    # Load and process knowledge graph
    kg = load_knowledge_graph(data_path, amazon_dataset_name)
    list_ent = list(set(kg['ent'].unique()) - set(['about', 'by', 'of', 'for']))
    
    # Initialize entity detector
    detector = EntityMentionDetector(list_ent, ngram_size=3, threshold=0.7)
    
    # Process common files first (files that don't depend on the split)
    process_entity_mappings(kg, metadata, items, output_path, data_path)
    
    # Process each split
    for split in splits:
        print(f"Processing {split} split...")
        data_file_name = f'{amazon_dataset_name}_llama3.1_reserved_special_token_10_{split}.jsonl'
        
        # Load and process the split-specific data
        df = pd.read_json(os.path.join(data_path, data_file_name), lines=True)
        
        # Parse conversations
        df.loc[:, "conversation"] = df["text"].swifter.apply(parse_conversation)
        
        # Replace names with IDs
        df = df.swifter.apply(replace_name_with_id, metadata=metadata, axis=1)
        
        # Process entity mentions
        results = detector.process_conversations(
            df.loc[:, 'conversation'].tolist(),
            batch_size=25,
            num_processes=None,
            cache_file=f"./entity_detection_cache_{split}.pkl"
        )
        results = [[x for x in result if x['role'] != 'system'] for result in results]
        
        if config_tokenizer == 'gpt2':
            process_gpt2(metadata, results, output_path, split)
        elif config_tokenizer == 'bert':
            process_bert(metadata, results, output_path, split)
        elif config_tokenizer == 'nltk':
            process_nltk(metadata, results, output_path, split)
        else:
            raise ValueError(f"Unsupported tokenizer configuration: {config_tokenizer}")


def get_item_sampled(row):
    """Extract the last 5 items from a user's history"""
    row['timestamp'], row['rating'], row['asin'], row['title'], row['text'] = zip(*sorted(
        zip(row['timestamp'], row['rating'], row['asin'], row['title'], row['text'])
    ))
    items = row['asin']
    return list(set(list(items)))


def parse_conversation(conversation_text):
    """
    Parse a conversation that follows the chat template format and extract role and content pairs.
    """
    pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>'
    matches = re.findall(pattern, conversation_text, re.DOTALL)
    
    messages = []
    for role, content in matches:
        role = role.strip()
        content = content.strip()
        messages.append({
            "role": role,
            "content": content
        })
    
    return messages


def load_metadata(data_path, dataset_name, num_threads=8) -> Dict[str, str]:
    """Load metadata from the McAuley-Lab/Amazon-Reviews-2023 dataset."""
    item_df = pd.read_pickle(os.path.join(data_path, f"{dataset_name}_item_df_core12.pkl"))
    raw_item_dict = item_df.set_index('parent_asin').to_dict()['Title']
    # Clean titles and format with ID tags
    item_dict = {
        f"<id_{key}>": re.sub(r"[\(\[].*?[\)\]]", "", str(value))
        for key, value in raw_item_dict.items()
    }
    
    print(f"Loaded metadata for {len(item_dict)} items")
    return item_dict

def replace_name_with_id(row, metadata):
    """Replace item names with their corresponding IDs in conversations"""
    names_items = {metadata[asin]: asin for asin in row['named_items']}
    target_items = {metadata[asin]: asin for asin in row['target']}
    
    for message in row['conversation']:
        if message['role'] in 'user':
            for name, item in names_items.items():
                message['content'] = message['content'].replace(name, item)
        else:
            for name, item in target_items.items():
                message['content'] = message['content'].replace(name, item)
    
    return row


def load_knowledge_graph(data_path, amazon_dataset_name):
    """Load and preprocess the knowledge graph data"""
    kg = pd.read_csv(
        os.path.join(data_path, "Movies_and_TV_KG", "cast_genre_mapping.csv"), 
    )
    
    kg['ent'] = kg['name']
    
    return kg


def format_CRSLab(conversation, metadata, tokenizer):
    """Format conversation data for CRSLab format"""
    mapping_role = {'user': 'Seeker', 'assistant': 'Recommender'}
    id_pattern = re.compile(r'<id_([^<>]+)>')
    dialog = []
    already_seen = set()
    
    for i, message in enumerate(conversation):
        movies = id_pattern.findall(message['content'])
        movies = [metadata.get("<id_"+movie+'>', "<id_"+movie+'>').strip() for movie in movies]
        message['content'] = message['content'].replace('<|reserved_special_token_10|>', '') 
        message['content'] = re.sub(r"<id_(.*?)>", r"\1", message['content']).strip()
        if message['role'] == 'assistant':
            movies = [movie for movie in movies if movie not in already_seen]
        already_seen.update(movies)
        dialog.append({
            'utt_id': i,
            "role": mapping_role[message['role']],
            "movies": movies,
            "entity": [x[0] for x in message['mentions']],
            "word": message['content'].split(),
            "text": get_tokens_only(message['content'], tokenizer)
        })
    
    return dialog


def get_tokens_only(text, tokenizer):
    """Convert text to tokens using the tokenizer"""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    return tokens


def obtain_int(text):
    """Convert text to a positive integer hash"""
    has_int = int(hash(text))
    return abs(has_int)


def process_entity_mappings(kg, metadata, items, output_path, data_path):
    """Process and save entity mappings and knowledge graph data"""
    # Process all items
    all_items_ = set(list(itertools.chain(*items.tolist())))
    all_items = pd.DataFrame([{"id": id_, "title": metadata['<id_'+id_+'>']} for id_ in all_items_])
    
    # Process link data
    link = process_link_data(kg, all_items, metadata, output_path, data_path)
    # Save movie IDs
    # movie_ids = list({title: obtain_int(title) for title in metadata.values()}.values())
    movie_ids = {id_: obtain_int(id_) for id_ in all_items_}
    with open(os.path.join(output_path, 'movie_ids.json'), 'w') as f:
        json.dump(movie_ids, f)
    
    # Update concept2id.json if it exists, otherwise create it
    try:
        with open(os.path.join(output_path, 'concept2id.json'), 'r') as f:
            concept2id = json.load(f)
        concept2id = {
            **{k: v for k, v in concept2id.items() if '@' not in k}, 
            **{id_: obtain_int(title) for id_, title in metadata.items()}
        }
    except FileNotFoundError:
        concept2id = {id_: obtain_int(title) for id_, title in metadata.items()}
    

    with open(os.path.join(output_path, 'concept2id.json'), 'w') as f:
        json.dump(concept2id, f)


def process_link_data(kg, all_items, metadata, output_path, data_path):
    """Process link data and create knowledge graph triples"""
    link = pd.read_csv(
        os.path.join(data_path, "reticsKG", "triples.csv"), 
    )
    
    all_items = all_items.rename(columns={"id": "id", "title": "name"})
    kn_mapping = pd.concat([kg, all_items], axis=0, ignore_index=True)
    # creact list of dict ID:name
    kn_mapping = kn_mapping.set_index('id').to_dict()['name']

    with open(os.path.join(output_path, 'entity2id.json'), 'w') as f:
        json.dump(kn_mapping, f)
    
    # Create triples
    link = link.groupby('head_id').agg(list).reset_index()
    link['triples'] = link.swifter.apply(format_triples, axis=1)
    link['head_id'] = link['head_id'].astype(str)
    link.set_index('head_id', inplace=True)
    
    dbpedia_subkg = link.to_dict()['triples']
    with open(os.path.join(output_path, 'dbpedia_subkg.json'), 'w') as f:
        json.dump(dbpedia_subkg, f)
    
    return link


def format_triples(row):
    """Format triples for knowledge graph"""
    entiti_rel = []
    for rel, ent in zip(row['relation_id'], row['tail_id']):
        entiti_rel.append([rel, ent])
    return entiti_rel


if __name__ == "__main__":
    # Configuration
    config_tokenizer = 'gpt2'  # or 'bert'
    data_path = "./data/distillrecdial"
    amazon_dataset_name = 'Movies_and_TV'
    output_path = f"/home/petruzzellia/apetruzz/llmDataset/convertToBaseline/strategy_{config_tokenizer}/"
    
    # Process all splits
    splits = ['train', 'test', 'valid']
    process_data(data_path, output_path, amazon_dataset_name, splits, config_tokenizer)

    # zip the output files
    import shutil
    shutil.make_archive(f"distillrecdial_{config_tokenizer}", 'zip', output_path)