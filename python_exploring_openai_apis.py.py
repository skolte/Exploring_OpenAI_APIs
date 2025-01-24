# %% [markdown]
# # Exploring OpenAI API

# %% [markdown]
# ## Task 1: Import Modules and Packages

# %%
from openai import OpenAI
import pandas as pd
import requests
from datetime import datetime
from pprint import pprint 
import tiktoken
from pypdf import PdfReader 
from IPython.display import Markdown, display 
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
import requests



# %% [markdown]
# ## Task 2: Set the API Key

# %%
client = OpenAI(api_key="YOUR API KEY HERE")

# %% [markdown]
# ## Task 3: Generate Emails for Reviews

# %%
columns = ['reviews', 'emails']
df = pd.DataFrame(columns=columns)
df['reviews'] = [
    "Nice socks, great colors, just enough support for wearing with a good pair of sneakers.",
    "Love Deborah Harness's Trilogy! Didn't want the story to end and hope they turn this trilogy into a movie. I would love it if she wrote more books to continue this story!!!",
    "SO much quieter than other compressors. VERY quick as well. You will not regret this purchase.",
    "Shirt a bit too long, with heavy hem, which inhibits turning over. I cut off the bottom two inches all around, and am now somewhat comfortable. Overall, material is a bit too heavy for my liking.",
    "The quality on these speakers is insanely good and doesn't sound muddy when adjusting bass. Very happy with these.",
    "Beautiful watch face. The band looks nice all around. The links do make that squeaky cheapo noise when you swing it back and forth on your wrist which can be embarrassing in front of watch enthusiasts. However, to the naked eye from afar, you can't tell the links are cheap or folded because it is well polished and brushed and the folds are pretty tight for the most part. love the new member of my collection and it looks great. I've had it for about a week and so far it has kept good time despite day 1 which is typical of a new mechanical watch."
]
df.head()

# Define the chat and postfix
chat = [{"role": "system", "content": "You are a polite representative."}]
postfix = "\n\nWrite an email to customers to address the issues put forward in the above review, thank them if they write good comments, and encourage them to make further purchases. Do not give promotion codes or discounts to the customers. Do not recommend other products. Keep the emails short."

# Define the function to generate email content
def getMail(review):
    chat_history = chat.copy()
    chat_history.append({"role": "user", "content": review + postfix})

    try:
        reply = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )
        return reply.choices[0].message.content
    except Exception as e:
        return f"Error generating email: {e}"

# Define the pretty print function
def pp(df):
    return display(
        df.style.set_properties(
            subset=['emails'],
            **{'text-align': 'left', 'white-space': 'pre-wrap', 'width': '900px'}
        )
    )

# Apply the getMail function to the reviews column
df['emails'] = df['reviews'].apply(getMail)

# Print the updated dataframe with pretty print
pd.set_option('display.max_colwidth', None)
pp(df)

# %% [markdown]
# ## Task 4: Generate Python Code

# %%
problems = [
    "primality testing",
    "sum of unique elements",
    "longest palindrome",
    "all possible permutations of a string",
]

# %%
columns = ['problems', 'answers']
df = pd.DataFrame(columns=columns)
df['problems'] = problems
# Convert the problems column values to uppercase before generating the Python code.
df['problems'] = [problem.upper() for problem in problems]  # Convert problems to uppercase

# Define the chat and postfix
chat = [{"role": "system", "content": "You are a good programmer."}]
postfix = "\n\nWrite python code to solve the specified problem statement. Please include comments and explanation where necessary. Make the code look pretty, and use clear, understandable names for functions, parameters and variables."

# Define the function to generate python code
def generatePythonCode(problem):
    chat_history = chat.copy()
    chat_history.append({"role": "user", "content": problem + postfix})

    try:
        reply = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )
        return reply.choices[0].message.content
    except Exception as e:
        return f"Error generating python code: {e}"

# Apply the generatePythonCode function to each problem in 'problems' column
# Generate answers and display in Markdown
for problem in df['problems']:
    code_response = generatePythonCode(problem)
    display(Markdown(f"### Problem: {problem}\n\n{code_response}"))

# Store responses in DataFrame
df['answers'] = df['problems'].apply(generatePythonCode)

# %% [markdown]
# ## Task 5: Summarize Text

# %%
# Function to calculate the number of tokens from a string
def num_tokens_from_string(text, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

# Download the PDF from the URL
url = "https://arxiv.org/pdf/1706.03762.pdf"
ppr_data = requests.get(url).content

# Save the PDF locally
with open('paper.pdf', 'wb') as handler:
    handler.write(ppr_data)

# Read and extract text from the first 2 pages of the PDF
reader = PdfReader("paper.pdf")
text = ""
for page in reader.pages[:2]:
    text += page.extract_text() + "\n"

# Get the token count for the extracted text
encoding = tiktoken.get_encoding('cl100k_base')
numberOfTokens = len(encoding.encode(text))
print(f"Number of Tokens: {numberOfTokens}")

# Define the maximum token length
maxTokenLength = 15384

# If token count exceeds the limit, trim the tokens and decode back to text
if numberOfTokens > maxTokenLength:    
    tokens = encoding.encode(text)  # Get encoded tokens
    trimmed_tokens = tokens[:maxTokenLength]  # Slice to max length
    text = encoding.decode(trimmed_tokens)  # Decode back to text

print("Final text (trimmed if necessary):")   
print(text[:500])  # Print first 500 characters of the trimmed text

# %%
# Define the chat and postfix
chat = [{"role": "system", "content": "You are an experienced Machine Learning research writer.."}]
postfix = "\n\nSummarize the above research paper in 1000 words."

# Define the function to summarize text
def summarizeText(text):
     # Tokenize the text again (if needed)
    encoding = tiktoken.get_encoding('cl100k_base')
    tokens = encoding.encode(text)

    # Append the trimmed text + postfix to chat history
    chat_history = chat.copy()
    chat_history.append({"role": "user", "content": text + postfix})

    try:
        reply = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )
        return reply.choices[0].message.content
    except Exception as e:
        return f"Error generating python code: {e}"

# Call the function to generate the summary
summary = summarizeText(text)

# Display the result using Markdown
display(Markdown(summary))

# %% [markdown]
# ## Task 6: Generate Images

# %%
for i in range(1):
    response = client.images.generate(
      model="dall-e-3",
      prompt="technical line drawing of the f-35 5th generation stealth fighter. Show and highlight a single powerful turbo jet engine. Show a large slightly angled view and smaller front and side views.",
      size="1792x1024",
      quality="hd",
      n=1,
    )
display(Markdown(response.data[0].revised_prompt))

image_url = response.data[0].url
path='usercode/images'
os.makedirs(path, exist_ok=True) 

name = path+'/Dall-e-generated-'+str(datetime.now())
img_data = requests.get(image_url).content

with open(name+'.jpg', 'wb') as handler:
    handler.write(img_data)

plt.figure(figsize=(11,9))
img = mpimg.imread(name+'.jpg')

imgplot = plt.imshow(img)
imgplot.axes.get_xaxis().set_visible(False)
imgplot.axes.get_yaxis().set_visible(False)
plt.show()

# %%


# %% [markdown]
# # End


