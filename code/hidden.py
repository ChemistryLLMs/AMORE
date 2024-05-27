import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import pandas as pd
from tqdm import tqdm
import pickle
import faiss

def save_hidden_states_model_decoder(data, model, tokenizer, path_to_res, prompt):  
    model.eval()
    batch_size = 1 

    t5_condgen_input_ids = []
    for i in tqdm(range(0, len(data['SMILES']), batch_size)):
        encoded_dict_t5condgen = tokenizer(list(prompt + i for i in data['SMILES'][i:i+batch_size]), return_tensors="pt", padding=False,truncation=True)
        t5_condgen_input_ids.append(encoded_dict_t5condgen['input_ids'])


    with torch.no_grad():
        t5_condgen_result = []

        for batch in tqdm(t5_condgen_input_ids):
            output = model(input_ids=batch,   
                        decoder_input_ids=batch,                                      
                        return_dict=True, 
                        output_hidden_states=True).decoder_hidden_states
            one_mol = []
            for t in output:
                t = torch.mean(t, dim=1)
                one_mol.append(t)
            t5_condgen_result.append(one_mol)

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)

def save_hidden_states_encoder(data, model, tokenizer, path_to_res, prompt):  
    batch_size = 1 

    t5_condgen_input_ids = []
    for i in tqdm(range(0, len(data['SMILES']), batch_size)):
        encoded_dict_t5condgen = tokenizer(list(prompt + i for i in data['SMILES'][i:i+batch_size]), return_tensors="pt", padding=False,truncation=True)
        t5_condgen_input_ids.append(encoded_dict_t5condgen['input_ids'])


    with torch.no_grad():
        t5_condgen_result = []

        for batch in tqdm(t5_condgen_input_ids):
            output = model.encoder(input_ids=batch, 
                                            return_dict=True, 
                                            output_hidden_states=True).hidden_states
            one_mol = []
            for t in output:
                t = torch.mean(t, dim=1)
                one_mol.append(t)
            t5_condgen_result.append(one_mol)

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)

def save_hidden_states_model(data, model, tokenizer, path_to_res, prompt):  
    model.eval()
    batch_size = 1 

    t5_condgen_input_ids = []
    for i in tqdm(range(0, len(data['SMILES']), batch_size)):
        encoded_dict_t5condgen = tokenizer(list(prompt + i for i in data['SMILES'][i:i+batch_size]), return_tensors="pt", padding=False,truncation=True)
        t5_condgen_input_ids.append(encoded_dict_t5condgen['input_ids'])


    with torch.no_grad():
        t5_condgen_result = []

        for batch in tqdm(t5_condgen_input_ids):
            output = model(input_ids=batch,                                       
                        return_dict=True, 
                        output_hidden_states=True).hidden_states
            one_mol = []
            for t in output:
                t = torch.mean(t, dim=1)
                one_mol.append(t)
            t5_condgen_result.append(one_mol)

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)

def save_hidden_states_scifive_decoder(data, model, tokenizer, path_to_res):  
    model.eval()
    with torch.no_grad():
        t5_condgen_result = []
        for text in tqdm(data['SMILES']):
            encoding = tokenizer.encode_plus(text + " </s>", pad_to_max_length=False, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
            output = model(
                input_ids=input_ids, attention_mask=attention_masks,
                decoder_input_ids=input_ids,
                return_dict=True,
                output_hidden_states=True).decoder_hidden_states
            one_mol = []
            for t in output:
                t = torch.mean(t, dim=1)
                one_mol.append(t)
            t5_condgen_result.append(one_mol)

    with open(path_to_res + '_hidden.pkl', 'wb') as f:
        pickle.dump(t5_condgen_result, f)
    
    return t5_condgen_result

def save_hidden_states_scifive_encoder(data, model, tokenizer, path_to_res):  
    model.eval()
    with torch.no_grad():
        t5_condgen_result = []
        for text in tqdm(data['SMILES']):
            encoding = tokenizer.encode_plus(text + " </s>", pad_to_max_length=False, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
            output = model.encoder(
                input_ids=input_ids, attention_mask=attention_masks,
                return_dict=True,
                output_hidden_states=True).hidden_states
            one_mol = []
            for t in output:
                t = torch.mean(t, dim=1)
                one_mol.append(t)
            t5_condgen_result.append(one_mol)

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
    topn = 1
    for i, v in enumerate(orig_data):
        #D, I = index.search(v.detach().numpy(), topn)
        D = index.search(v.detach().numpy(), topn)  # Возвращает результат: Distances, Indices
        print(D)
        dist.append(D[0][0])
        ind.append(I[0][0])
        trueres.append(I[0][0] == i)

    return dist, ind, trueres

def count_hidden(test_data_path, orig_data_path):
    with open(orig_data_path + '_hidden.pkl', 'rb') as f:
        orig_data = pickle.load(f)

    with open(test_data_path + '_hidden.pkl', 'rb') as f:
        test_data = pickle.load(f)

    return [sum(create_dist([orig_data[i][u] for i in range(3300)], [test_data[i][u]for i in range(3300)])[-1]) for u in range(len(orig_data[0]))]