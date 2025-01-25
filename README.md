# Exploring OpenAI API

This repository contains simple and practical tasks to demonstrate the capabilities of OpenAI APIs, including GPT-3.5 and DALL-E. Through these tasks, I explored various functionalities such as email generation, Python code generation, text summarization, and image generation. These examples aim to show how OpenAI APIs can be used to automate repetitive tasks, enhance productivity, and generate creative outputs.

---

## Features and Tasks

### 1. **Email Generation Based on Amazon Reviews**
   - **Objective**: Generate personalized emails addressing customer concerns, thanking them for their purchase, and encouraging them to shop again.  
   - **Scenario**: Instead of manually crafting responses, GPT-3.5 is used to automate email generation for customer reviews.  
   - **Output**: Polite and relevant emails tailored to each review.  

### 2. **Python Code Generation**
   - **Objective**: Generate Python code to solve common algorithmic problems.  
   - **Sample Problems**:
     - Primality Testing
     - Sum of Unique Elements
     - Longest Palindrome
     - All Possible Permutations of a String  
   - **Approach**: Use GPT-3.5 to automate coding solutions with comments and readable function structures.

### 3. **Text Summarization**
   - **Objective**: Analyze and summarize complex research papers for better accessibility.  
   - **Example**: Generated a concise summary of the research paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).

### 4. **Image Generation**
   - **Objective**: Generate a technical line drawing of an F-35 fighter jet.  
   - **Tool**: DALL-E 3 was used to produce high-quality visuals, showcasing its potential in creative and technical applications.  

---

## Things I Learned

### **API Integration**
- Successfully integrated OpenAI APIs to handle structured and unstructured data, enabling effective model prompting and interaction.

### **Data Manipulation with Pandas**
- Used Pandas for advanced data handling, ensuring seamless management of diverse datasets.

### **Text Processing and Tokenization**
- Explored tokenization and text encoding using the `tiktoken` library to optimize model inputs.
- Utilized the `pypdf` library to extract and process text from PDFs for further analysis.

### **Dynamic Content Generation**
- **Sentiment Analysis**: Generated personalized email responses based on customer reviews.
- **Code Generation**: Automated problem-solving tasks with Python code created by GPT-3.5.
- **Text Summarization**: Summarized research papers for improved comprehension and accessibility.

### **Image Generation and Visualization**
- Created high-quality visuals using OpenAI’s DALL-E model.
- Managed image processing tasks (e.g., saving and displaying visuals) using `matplotlib` for professional presentations.

### **Task Automation**
- Developed reusable functions (e.g., `generateMail`, `generatePythonCode`, `summarizeText`) to automate repetitive workflows, enhancing efficiency and scalability.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/skolte/ExploreOpenAiAPIs.git
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in the appropriate script or environment variable.
4. Run individual scripts to explore the various tasks:
   - Email generation
   - Python code generation
   - Text summarization
   - Image generation

---

