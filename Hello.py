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



def run():
    
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )
    # Import required libraries
    #import streamlit as st
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Initialize the GPT-2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Create the Streamlit web app
    st.title("Chat with GPT-2")

    # Create a text input for the user
    user_input = st.text_input("You: ", "")

    # Generate a response from the model if the user enters input
    if user_input:
        # Encode the user's input and generate a response
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        chat_output = model.generate(
            new_user_input_ids, 
            max_length=100, 
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=5,  # Generate 5 responses
            num_beams=5  # Use beam search with 5 beams
        )
        
        # Decode and display each of the model's outputs
        for i, output in enumerate(chat_output):
            chat_output_text = tokenizer.decode(output[new_user_input_ids.shape[-1]:], skip_special_tokens=True)
            st.write(f"Response {i+1}: {chat_output_text}")

            # Collect user feedback
            feedback_up = st.button(f"👍 Thumbs Up for Response {i+1}")
            feedback_down = st.button(f"👎 Thumbs Down for Response {i+1}")
            
            if feedback_up:
                st.write("Thanks for your feedback!")
                logging.info(f"User Input: {user_input} | Model Response: {chat_output_text} | Feedback: Thumbs Up")

            if feedback_down:
                st.write("Thanks for your feedback!")
                logging.info(f"User Input: {user_input} | Model Response: {chat_output_text} | Feedback: Thumbs Down")

if __name__ == "__main__":
    run()
