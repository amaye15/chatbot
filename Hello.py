# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = get_logger(__name__)
# Initialize the logger
import logging
logging.basicConfig(filename='feedback.log', level=logging.INFO, format='%(asctime)s %(message)s')

import requests

def query(payload, model_id, api_token = "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"):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json= payload )
	return response.json()



def run():
    
    st.set_page_config(
        page_title="Demo",
        page_icon="👋",
        layout="wide"
    )

    # Create the Streamlit web app
    st.title("Chat with GPT-2")

    model_id = "gpt2"
    #api_token = "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" # get yours at hf.co/settings/tokens

    # Initialize the GPT-2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create a text input for the user
    user_input = st.text_input("You: ", "")

    #data = query(user_input, model_id, api_token)

    # Create columns for response text and feedback buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    temperatures = [1.0, 10.0, 100.0]
    with st.form(key='my_form'):
    # Generate a response from the model if the user enters input
        if user_input:
        
            for i, temp in enumerate(temperatures):
                # Encode the user's input and generate a response
                new_user_input_ids = tokenizer.encode(tokenizer.eos_token + user_input, return_tensors='pt')

                chat_output = model.generate(
                    new_user_input_ids, 
                    max_length=100, 
                    pad_token_id=tokenizer.eos_token_id,
                    #num_return_sequences=5,  # Generate 5 responses
                    #num_beams=5,  # Use beam search with 5 beams
                    temperature=temp,  # Different temperature values for diversity
                    do_sample = True,
                    no_repeat_ngram_size=1)
                
                # Decode and display each of the model's outputs
                for output in chat_output:
                    chat_output_text = tokenizer.decode(output[new_user_input_ids.shape[-1]:], skip_special_tokens=True)
                    
                    st.write(f"Response {i+1}: {chat_output_text}")

                    # Collect user feedback using a slider
                    rating = st.slider(f"Rate Response {i+1}", min_value=1, max_value=10, value=1)

                    # Collect user feedback
                    #feedback_up = st.button(f"👍 Thumbs Up for Response {i+1}")
                    #feedback_down = st.button(f"👎 Thumbs Down for Response {i+1}")
                    
                    #if col2.button(f"👍 {i+1}"):
                        #st.write("Thanks for your feedback!")
                        #logging.info(f"User Input: {user_input} | Model Response: {chat_output_text} | Feedback: Thumbs Up")

                    #if col3.button(f"👎 {i+1}"):
                        #st.write("Thanks for your feedback!")
                        #logging.info(f"User Input: {user_input} | Model Response: {chat_output_text} | Feedback: Thumbs Down")
        submit_button = st.form_submit_button(label='Submit')
        

if __name__ == "__main__":
    run()
