import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


#function to get responce from the Llama2


def getLlamaResponce(input_text,no_words,blog_style):
    
    # Load LLama2 model
    llm = CTransformers(model='D:\Codding\Artifitial Intelligence\LLMs\Blog Generation LLM\models\llama-2-7b-chat.ggmlv3.q8_0.bin', 
                          model_type='llama',
                          config={'max_new_tokens': 256,
                                  'temperature': 0.01,})
    
    # Load the prompt template
    template="""
    write a blog on the topic: {input_text} for {blog_style} with {no_words} words
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
   
    # Generate the response from Llama2
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response





st.set_page_config(page_title="Generate Blogs using Llama2 🤖",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')



st.header("Generate Blogs using Llama2 🤖")

input_text=st.text_input("Enter the Blog Topic")


## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No. of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")


#Final response to frontend
if submit:
    st.write(getLlamaResponce(input_text,no_words,blog_style))
    
