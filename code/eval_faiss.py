import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import faiss

def save_last_hidden_state_model(data, model, tokenizer, path_to_res):  
    batch_size = 1 

    t5_condgen_input_ids = []
    for i in tqdm(range(0, len(data['SMILES']), batch_size)):
        encoded_dict_t5condgen = tokenizer(list(data['SMILES'][i:i+batch_size]), 
                                           return_tensors="pt", padding=False, truncation=True)
        t5_condgen_input_ids.append(encoded_dict_t5condgen['input_ids'])


    with torch.no_grad():
        t5_condgen_result = []
        for batch in tqdm(t5_condgen_input_ids):
            output = model(input_ids=batch, return_dict=True).last_hidden_state
            t5_condgen_result.append(torch.mean(output, dim=1))

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)

    return t5_condgen_result


def save_last_hidden_state_encoder(data, model, tokenizer, path_to_res):  
    batch_size = 1 

    t5_condgen_input_ids = []
    for i in tqdm(range(0, len(data['SMILES']), batch_size)):
        encoded_dict_t5condgen = tokenizer(list(data['SMILES'][i:i+batch_size]), 
                                           return_tensors="pt", padding=False, truncation=True)
        t5_condgen_input_ids.append(encoded_dict_t5condgen['input_ids'])

    with torch.no_grad():
        t5_condgen_result = []
        for batch in tqdm(t5_condgen_input_ids):
            output = model.encoder(input_ids=batch, return_dict=True).last_hidden_state
            t5_condgen_result.append(torch.mean(output, dim=1))

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)
    
    return t5_condgen_result


def create_dist(test_data, orig_data):

    index = faiss.IndexFlatL2(len(test_data[0][0]))
    for v in test_data:
        index.add(v.detach().numpy())
    dist = []
    ind = []
    trueres = []
    top5 = []
    topn = 5
    for i, v in enumerate(orig_data):
        D, I = index.search(v.detach().numpy(), topn)  # Возвращает результат: Distances, Indices
        dist.append(D[0][0])
        ind.append(I[0][0])
        trueres.append(I[0][0] == i)
        top5.append(i in I[0])

    return dist, ind, trueres, top5

def count_res(test_data_path, orig_data_path):
    with open(orig_data_path + '_hidden.pkl', 'rb') as f:
        orig_data = pickle.load(f)

    with open(test_data_path + '_hidden.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    [sum(create_dist([orig_data[i][u] for i in range(3300)], [test_data[i][u]for i in range(3300)])[-1]) for u in range(len(orig_data[0]))]
    return [sum(create_dist([orig_data[i][u] for i in range(3300)], [test_data[i][u]for i in range(3300)])[-1]) for u in range(len(orig_data[0]))]

def save_last_hidden_state_bert(data, model, tokenizer, path_to_res):  
    with torch.no_grad():
        t5_condgen_result = []
        for batch in tqdm(data):
            inputs = tokenizer(batch, return_tensors="pt", return_token_type_ids=False, padding=False, truncation=True, max_length=128)
            output = model(**inputs).last_hidden_state
            t5_condgen_result.append(torch.mean(output, dim=1))

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)
    
    return t5_condgen_result

def save_last_hidden_state_bart(data, model, tokenizer, path_to_res):  
    with torch.no_grad():
        t5_condgen_result = []
        for batch in tqdm(data):
            inputs = tokenizer(batch, return_tensors="pt", return_token_type_ids=False, padding=False, truncation=True, max_length=128)
            output = model.encoder(**inputs).last_hidden_state
            t5_condgen_result.append(torch.mean(output, dim=1))

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)
    
    return t5_condgen_result

def save_last_hidden_state_scifive(data, model, tokenizer, path_to_res):  
    model.eval()
    with torch.no_grad():
        t5_condgen_result = []
        for text in tqdm(data):
            encoding = tokenizer.encode_plus(text + " </s>", pad_to_max_length=False, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
            output = model.encoder(input_ids=input_ids, attention_mask=attention_masks,
                    return_dict=True).last_hidden_state
            t5_condgen_result.append(torch.mean(output, dim=1))

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)
    
    return t5_condgen_result

def create_dist_MRR(test_data, orig_data):

    index_v = faiss.IndexFlatL2(len(test_data[0][0]))
    for v in test_data:
        index_v.add(v.detach().numpy())
    dist = []
    ind = []
    trueres = []
    top5 = []
    topn = 3300
    range_of_trueres = []
    for i, v in enumerate(orig_data):
        D, I = index_v.search(v.detach().numpy(), topn)  # Возвращает результат: Distances, Indices
        dist.append(D[0][0])
        ind.append(I[0][0])

        trueres.append(I[0][0] == i)
        top5.append(i in I[0])
        range_of_trueres.append(list(I[0]).index(i))
        #print(range_of_trueres)

    return dist, ind, trueres, top5, range_of_trueres
