import streamlit as st
from transformers import pipeline
from transformers.tokenization_utils import TruncationStrategy

import tokenizers
import pandas as pd
import requests

st.set_page_config(
     page_title='ScrollBERT Demo',
     page_icon="📜",
     initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>

    .sidebar .sidebar-content {
        background-image: linear-gradient(#3377ff,  #80aaff);
    }

    footer {
        color:white;
        visibility: hidden;
    }
    input {
        direction: rtl;
    }
    .stTextInput .instructions {
        color: grey;
        font-size: 9px;}

</style>
<div style="color:white; font-size:13px; font-family:monospace;position: fixed; z-index: 1; bottom: 0; right:0; background-color: #f63766;margin:3px;padding:8px;border-radius: 5px;"><a href="https://huggingface.co/onlplab/alephbert-base"  target="_blank" style="text-decoration: none;color: white;">Use aleph-bert in your project </a></div>
""",
    unsafe_allow_html=True,
)

    
    

@st.cache(show_spinner=False)
def get_json_from_url(url):
    return requests.get(url).json()

# models = {"alephbert": ("./alephbert/alephbert_40_epochs", "./alephbert/alephbert")}
models = {"ScrollBERT": ("./alephbert/alephbert_40_epochs", "onlplab/alephbert-base")}
# models = {"alephbert": (r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert_40_epochs", r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert")}

@st.cache(show_spinner=False, hash_funcs={tokenizers.Tokenizer: str})
def load_model(model):
    pipe = pipeline('fill-mask', models[model][0], tokenizer=models[model][1])
    def do_tokenize(inputs):
        return pipe.tokenizer(
                inputs,
                add_special_tokens=True,
                return_tensors=pipe.framework,
                padding=True,
                truncation=TruncationStrategy.DO_NOT_TRUNCATE,
            )

    def _parse_and_tokenize(
        inputs, tokenized=False, **kwargs
    ):
        if not tokenized:
            inputs = do_tokenize(inputs)
        return inputs

    pipe._parse_and_tokenize = _parse_and_tokenize
    
    return pipe, do_tokenize



import os


st.title('ScrollBERT📜')
st.sidebar.markdown(
    """<div>
      <p style="color:white; font-size:13px; font-family:monospace; text-align: center">ScrollBERT Demo &bull; </p></div>
      <br>""",
    unsafe_allow_html=True,
)

mode = 'Models'

if mode == 'Models':
    model = st.sidebar.selectbox(
     'Select Model',
     list(models.keys()))
    masking_level = st.sidebar.selectbox('Masking Level:', ['Tokens', 'SubWords'])
    n_res = st.sidebar.number_input(
        'Number Of Results',
        format='%d',
        value=5,
        min_value=1,
        max_value=100)
    
    model_tags = model.split('-')
    model_tags[0] = 'Model:' + model_tags[0] 

    st.markdown(''.join([f'<span style="color:white; font-size:13px; font-family:monospace; background-color: #f63766;margin:3px;padding:8px;border-radius: 5px;">{tag}</span>' for tag in model_tags]),unsafe_allow_html=True)
    st.markdown('___')
    ####
    #prepare the model
    ####
    
    unmasker, tokenize = load_model(model)
    
    
    ####
    # get inputs
    ####
            
    input_text = st.text_input('Insert text you want to mask', '')
    if input_text:
        input_masked = None
        tokenized = tokenize(input_text)
        ids = tokenized['input_ids'].tolist()[0]
        subwords = unmasker.tokenizer.convert_ids_to_tokens(ids)
        
        if masking_level == 'Tokens':
            tokens = str(input_text).split()
            tokens_with_empty = [''] + tokens
            # masked_token = st.selectbox('Select token to mask:', [''] + tokens)
            masked_token_index = st.selectbox('Select token to mask:', range(len(tokens_with_empty)), format_func=lambda x:tokens_with_empty[x])
            masked_token_index -= 1
            if masked_token_index > -1:
                input_masked = ' '.join(tokens[i] if i != masked_token_index else '[MASK]' for i in range(len(tokens)))
                display_input = input_masked
        if masking_level == 'SubWords':
            tokens = subwords
            idx = st.selectbox('Select token to mask:', list(range(0,len(tokens)-1)), format_func=lambda i: tokens[i] if i else '')
            tokenized['input_ids'][0][idx] = unmasker.tokenizer.mask_token_id
            ids = tokenized['input_ids'].tolist()[0]
            display_input = ' '.join(unmasker.tokenizer.convert_ids_to_tokens(ids[1:-1]))
            if idx:
                input_masked = tokenized
                
        if input_masked: 
            st.markdown('#### Input:')
            ids = tokenized['input_ids'].tolist()[0]
            subwords = unmasker.tokenizer.convert_ids_to_tokens(ids)
            st.markdown(f'<p dir="rtl">{display_input}</p>',
                        unsafe_allow_html=True,
            )
            st.markdown('#### Outputs:')
            res = unmasker(input_masked, tokenized=masking_level == 'SubWords', top_k=n_res)
            if res:
                res = [{'Prediction':r['token_str'], 'Completed Sentence':r['sequence'].replace('[SEP]', '').replace('[CLS]', ''), 'Score':r['score']} for r in res]
                res_table = pd.DataFrame(res)
                st.table(res_table)
            
            

        








    





