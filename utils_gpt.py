from sqlalchemy import create_engine

import os
import email_db_utils as edu

from datetime import datetime, timedelta

import openai

import pickle
from getpass import getpass
from time import sleep


def get_email_data_(topic, start_date=None, end_date=None, sender=None, schema='market_color'):
    utilFolder = '//LSFS/BOSTON/Dept/FIM/HF2/ChatGPT/market_color_email_data'
    os.chdir(utilFolder)
    # Create an Instance of the DataProcessor class
    email_data = edu.DataProcessor()
    # Input Parameters Handling
    topic = topic.lower()
    if topic not in ['ig', 'hy', 'bl', 'em', 'ir', 'ec', 'eq']:
        print("Warning! Unexpected topic selected:", topic)
        return None
    if start_date is None:
        start_date = datetime.now() + timedelta(-7)
    if end_date is None:
        end_date = datetime.now() + timedelta(1)
    # Retrieve Emails
    table_dic = email_data.get_table_map()
    try:
        data_for_topic = email_data.query_email_table(schema, table_dic[topic], start_date, end_date)
    except KeyError:
        print("Warning! No emails found on the topic:", topic)
        return None
    if data_for_topic.empty:
        print("Warning! No emails found on the topic:", topic)
        return None
    if sender is None:
        return data_for_topic
    else:
        return data_for_topic[data_for_topic['sender'].str.constains(sender, case=False)]


def get_email_content_(row):
    if row['sender'].endswith('O=LoomisSayles'):
        if row['subject'].startswith('Fw: '):
            tmp_subject = row['subject'][4:]
            idx_start = row['body_text'].find(tmp_subject)
            if idx_start != -1:
                return tmp_subject + '\n' + row['body_text'][idx_start:]
    if row['sender'].startswith('"News Alert (BLOOMBERG)"'):
        tmp_starter = '(Bloomberg) -- '
        tmp_signature = 'To contact the reporter on this story: '
        idx_start = row['body_text'].find(tmp_starter)
        idx_end = row['body_text'].find(tmp_signature)
        if idx_start != -1 and idx_end != -1:
            idx_start += len(tmp_starter)
            return row['subject'] + '\n' + row['body_text'][idx_start:idx_end]
        else:
            return row['subject']
    return row['body_text']


def get_emails(topic, start_date=None, end_date=None):
    df = get_email_data_(topic, start_date, end_date)
    if df is not None:
        df['content']=df.apply(get_email_content_, axis=1, raw=False)
    return df


def get_openai_key():
    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        print('Enter OpenAI API Key (https://platform.openai.com/account/api-keys):')
        os.environ['OPENAI_API_KEY'] = getpass()
        openai.api_key = os.environ['OPENAI_API_KEY']


def run_gpt(prompt, short_answer=True, model='gpt-3.5-turbo-16k'):
    utilFolder = '//LSFS/BOSTON/Dept/FIM/HF2/ChatGPT/market_color_email_data'
    os.chdir(utilFolder)
    user = os.getenv("USERNAME")
    try:
        with open('saved_cache.pkl', 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}
        with open('saved_cache.pkl', 'wb') as f:
            pickle.dump(cache, f)
    cache_key = (prompt, short_answer, model)
    if cache_key in cache:
        return cache[cache_key]
    try:
        openai.organization = os.environ['OPENAI_API_ORG']
    except KeyError:
        os.environ['OPENAI_API_ORG'] = "org-l8AOyRijW30rxUerCRGW4NRT"
        openai.organization = os.environ['OPENAI_API_ORG']
    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        print('Enter OpenAI API Key (https://platform.openai.com/account/api-keys):')
        os.environ['OPENAI_API_KEY'] = getpass()
        openai.api_key = os.environ['OPENAI_API_KEY']
    if prompt is not None and len(prompt)>0:
        messages = [{"role": "system", "content": "You are a successful portfolio manager."}]
        messages += [{"role": "user", "content": prompt}]
    else:
        return None
    if short_answer:
        max_tokens = 618
    else:
        max_tokens = 888
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = 0,
                max_tokens = max_tokens,
                top_p = 1,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
            )['choices'][0]['message']['content'].strip()
            break
        except openai.error.RateLimitError:
            if i > 8: raise
            sleep(i*30)
    cache[cache_key] = response
    with open('saved_cache.pkl', 'wb') as f:
        pickle.dump(cache, f)
    return response


def run_chat_gpt(prompt, dialogs=None, short_answer=True, model='gpt-3.5-turbo-16k'):
    utilFolder = '//LSFS/BOSTON/Dept/FIM/HF2/ChatGPT/market_color_email_data'
    os.chdir(utilFolder)
    user = os.getenv("USERNAME")
    try:
        with open('saved_chats.pkl', 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}
        with open('saved_chats.pkl', 'wb') as f:
            pickle.dump(cache, f)
    cache_key = (prompt, short_answer, model, dialogs) # UNDER CONSTRUCTION
    if cache_key in cache:
        return cache[cache_key]
    try:
        openai.organization = os.environ['OPENAI_API_ORG']
    except KeyError:
        os.environ['OPENAI_API_ORG'] = "org-l8AOyRijW30rxUerCRGW4NRT"
        openai.organization = os.environ['OPENAI_API_ORG']
    try:
        openai.api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        print('Enter OpenAI API Key (https://platform.openai.com/account/api-keys):')
        os.environ['OPENAI_API_KEY'] = getpass()
        openai.api_key = os.environ['OPENAI_API_KEY']
    if dialogs is None:
        messages = [{"role": "system", "content": "You are a successful portfolio manager."}]
    else:
        messages = dialogs
    if prompt is not None and len(prompt)>0:
        messages += [{"role": "user", "content": prompt}]
    else:
        return None, messages
    if short_answer:
        max_tokens = 618
    else:
        max_tokens = 888
    for i in range(10):
        try:
            response = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = 0,
                max_tokens = max_tokens,
                top_p = 1,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
            )['choices'][0]['message']['content'].strip()
            break
        except openai.error.RateLimitError:
            if i > 8: raise
            sleep(i*30)
    dialogs = messages + [{"role": "assistant", "content": response}]
    cache[cache_key] = (response, dialogs)
    with open('saved_chats.pkl', 'wb') as f:
        pickle.dump(cache, f)
    return response, dialogs


"""
response = openai.Completion.create(
    model = 'curie',
    prompt = "hello world!",
    temperature = 0,
    max_tokens = 888,
    top_p = 1,
    frequency_penalty = 0.0,
    presence_penalty = 0.0,
)['choices'][0]['text'].strip()
"""


__all__ = ["get_emails", "run_gpt", "get_openai_key"]

