from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from translate import Translate
from csv import writer
import pandas as pd
import re
import datetime
import requests
from langdetect import detect, LangDetectException
import time
import spacy
import signal

nlp = spacy.load("xx_ent_wiki_sm")
global_llm_instace = None
error_encounter = False
session = requests.Session()
running = True
questions_file = r"../data/questions-answers.csv"

def is_english(text):
    try: 
        return detect(text) == "en"
    except LangDetectException: 
        return False
            
def load_data(file_path):
    loader = CSVLoader(
        file_path=file_path,
        encoding='utf-8',  
        csv_args={
            "delimiter": ",",
            "quotechar": '"'
        }
    )
    return loader.load()

def create_embeddings(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': "cpu"})
    return FAISS.from_documents(documents, embeddings)

def create_prompt_template():
    template = """
    <s> [INST] You are a helpful, respectful and honest customer support agent in our company. 
    Your task is to provide direct and helpful answers to customer questions. 
    Use the following context and chat to answer the question at the end.
    You should not generate answer for any unrelated question.
    Instead you should answer with company prepared static answer. Focus on not modifying, not adding anything or not changing the static response to unrelated questions.
    Static answer is below:
    I'm sorry but I can't answer your question. We always value and appreciate the feedback we receive from our customers and will use it to provide a better experience. 
    If you would like to contact our support team, you can send an email to support@gmail.com. Thank you for taking the time to send us your thoughts and opinions.

    Focus on answering only one long sentece without punctuation.
    Response should be very similar or even identical to the past responses, in terms of length, tone of voice, logical arguments, and other details.
    Context: {context}
    History: {history}
    Question: {question}
    Provide a single and direct response below: [/INST]
    """
    return PromptTemplate(template=template, input_variables=['context', 'history', 'question'])

def create_qa_llm():
    global global_llm_instace
    if global_llm_instace is not None:
        return global_llm_instace
    
    llm = CTransformers(
        model='mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        model_type='mistral',
        verbose=False,
        config={'max_new_tokens': 4096, 'temperature': 0.01, 
                'context_length': 8192, 'gpu_layers': 0}
    )
    
    db = create_embeddings(load_data("customer_response_data.csv"))
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    prompt = create_prompt_template()

    qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=retriever,
                                         return_source_documents=False,
                                         chain_type_kwargs={"verbose": False, 'prompt': prompt, 'memory': memory}
                                         )
    
    global_llm_instace = qa_llm
    return qa_llm

def has_question(questions_file: str, message: str) -> tuple[bool, str]:
    questions = pd.read_csv(questions_file)

    # Iterate through rows and check if message column contains question
    for _, row in questions.iterrows():
        if row["Customer Message"] == message:
            
            return (True, row["Response"])
        
    return (False, "")


def save_question(questions_file: str, message: str, answer: str) -> None:

    with open(questions_file, 'a', newline='') as file:
        write = writer(file)
        write.writerow([message, answer])

def extract_name(email_content):
    """
    Extract names from the emial, content using spacy NER 
    """
    ignore_terms = {"Sir", "Madam", "Madams", "Sirs", "Dear Sir", "Dear Sirs", "Dear Madam", "Dear Madams"}
    doc = nlp(email_content)
    names = [ent.text for ent in doc.ents if ent.label_ in ["PER", "PERSON"] and ent.text not in ignore_terms] 
    return names[-1] if names else None

# message = 'Dear Sir, I am having trouble with PayPal and I do not know how to withdraw money, can you tell me how to do that? Kind regards, John'
# message = 'Dear Sir, I am having trouble withdrawing money, can you tell me how should I do that and what the smallest amount I could withdraw? Thank you in advance, I am looking forward to hearing from you,  Kind regards, Alice'
# print(extract_name(message))
    
def get_greeting(name=None):

    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
    elif 12 <= current_hour < 18:
        greeting =  "Good afternoon"
    else:
        greeting =  "Good evening"
    
    greeting = f"{greeting}{', ' + name if name else ''}! <br><br>"
    return greeting        

def is_automatic_email(email):
    # Advanced keyword list
    keywords = [
        'do not reply', 'noreply', 'automatic email', 'auto-generated',
        'not monitored', 'subscription', 'out of office', 'system notification'
    ]
    
    # Header checks
    headers_to_check = ['precedence', 'auto-submitted', 'x-auto-response-suppress']
    for header, value in email.items():
        if header.lower() in headers_to_check and value.lower() != 'no':
            return True
    
    # Sender's email check
    sender = email.get('from', '').lower()
    if 'noreply' in sender or 'no-reply' in sender:
        return True
    
    # Subject and body checks with safe handling for None values
    subject = email.get('subject', '') or ''  # Ensure subject is a string
    body = email.get('message', '') or ''  # Ensure body is a string
    subject = subject.lower()
    body = body.lower()
    
    if any(keyword in subject or keyword in body for keyword in keywords):
        return True
    
    # Reply-To header check
    reply_to = email.get('reply-to', '').lower()
    if 'noreply' in reply_to or 'no-reply' in reply_to:
        return True
    
    # Passed all checks, not an automatic email
    return False

# creative_style_detected = re.search(r'\b(poem|rap|song|rhyme|haiku|limerick|sonnet|ballad|prose|narrative|monologue|dialogue|short story|joke|anecdote|riddle|free verse|acrostic|elegy|satire)\b', text, re.IGNORECASE)

def get_answer(qa_model, text):
    # Expanded regex to include more styles
    creative_style_detected = re.search(r'\b(poem|rap|song|rhyme|haiku|limerick|sonnet|ballad|prose|narrative|monologue|dialogue|short story|joke|anecdote|riddle|free verse|acrostic|elegy|satire)\b', text, re.IGNORECASE)

    if creative_style_detected:
        # Updated regex to handle additional styles
        text = re.sub(r'\b(in form of a |as a |like a )?(poem|rap|song|rhyme|haiku|limerick|sonnet|ballad|prose|narrative|monologue|dialogue|short story|joke|anecdote|riddle|free verse|acrostic|elegy|satire)\b', '', text, flags=re.IGNORECASE)
        text = "Please provide a direct and factual answer to the following question: " + text

    name = extract_name(text)
    greetings = get_greeting()
    output = qa_model.invoke({'query': text})
    
    answer_text = output.get('result', 'I am sorry, I cannot process your request at the moment.')

    # Standard greetings and closing statement
    closing_statement = "<br><br>Thank you for choosing our company! We're delighted to have you with us and appreciate your trust in our application. Have a nice day!"
    sign_off = "<br><br> Kind regards, <br>Chatbot<br><br>"
    full_response = f'{greetings}\n\n{answer_text}\n\n{closing_statement}\n\n{sign_off}'

    return full_response


def fetch_emails():
    global error_encounter
    url = 'http://localhost/get-email-to-response?mailbox=support'
    try:
        response = requests.get(url)
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            print(response.text)
            return response.json()
        else:
            print(f"Error fetching emails: Status code {response.status_code}, Content-Type {response.headers.get('Content-Type')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}. Exiting...")
        error_encounter = True
        return None

def skip_email_response(email_id):
    global error_encounter
    url = 'http://localhost/send-email-responded?mailbox=support'
    data = {
        'emailid_to': email_id,
        'reply': False,
    }
    headers = {'Content-Type': 'application/json'}
    response = session.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Skip response sent successfully")
    else:
        error_encounter = True
        print(f"Failed to send the response, status code {response.status_code}, response text: {response.text}. Exiting...")
        

def send_email_response(email_id, response_text, in_reply_to=None, references=None, subject="Response from Support", from_address=None):
    global error_encounter
    if from_address is None:
        raise ValueError("Recipient email address must be provided")
    
    if not subject:
        subject = "Chatbot Support Response"
    
    url = 'http://localhost/send-email-responded?mailbox=support'
    data = {
        'emailid_to': email_id,
        'in_reply_to': in_reply_to,
        'references': references,
        'subject': subject,
        'to': from_address,
        'message': response_text,
    }
    headers = {'Content-Type': 'application/json'}
    response = session.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Sent response:", response_text)
        print("Response sent successfully")
    else:
        error_encounter = True
        print(f"Failed to send the response, status code {response.status_code}, response text: {response.text}")

        
def has_email_been_replied(email_id):
    url = 'http://localhost/get-email-to-response?mailbox=support'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('replied', False)
    else:
        print(f"Failed to check if email {email_id} has been replied, status code: {response.status_code}.")
        return False

def process_emails(qa_llm):
    # last_email_id = get_last_email_id()
    start_time = time.time()  # Start timing
    emails_data = fetch_emails()
    translator = Translate()


    if emails_data and emails_data.get('success') and emails_data.get('data'):
        email = emails_data['data']
        email_id = email.get('emailid')
        email_message = email.get('message')
        subject = email.get('subject')
        source_message = email_message

        # print(f"Last processed email ID: {last_email_id}, Current email to be processed: {email_id}")
        print(f"Curently processing id: {email_id}")
        
        if has_email_been_replied(email_id) or is_automatic_email(email) or (email.get('in_reply_to') and not email.get('references')):
            skip_email_response(email_id)
            print(f"Email ID {email_id} has already been processed or is an automatic email, skipping.")
            return 
        
        if not is_english(email_message):
            # skip_email_response(email_id)
            # print(f"Email ID {email_id} is not in English, skipping.")
            # return  # Skip this email and leave it unread

            email_message_translated = translator.translate_text(email_message)
            email_message = email_message_translated["data"]["translations"][0]["translatedText"]
            print(email_message)
        
        print("Processing new, unopened email...")

        # Check if csv has question in source language
        question_check = has_question(questions_file, source_message)
        if question_check[0]:

            response_text = question_check[1]

        else:

            response_text = get_answer(qa_llm, email_message)

            response_text_translated = translator.translate_text()
            response_text = response_text_translated["data"]["translations"][0]["translatedText"]
            print(response_text)


            # Save q-a pair to csv
            save_question(questions_file, source_message, response_text)

        response_text = refactor_answer(response_text)

        from_address = email.get('from')
        send_email_response(email_id, response_text, email.get('in_reply_to'), email.get('references'), subject, from_address)
        end_time = time.time()  # End timing
        print(f"Generating the response took {end_time - start_time:.2f} seconds.")
    else: 
        print("No new emails or failed to fetch emails.")


def refactor_answer(input_text):
    refactored_answer = ""
    dash_list_regex = " -[^ ].*?\. "
    split_list = re.split(dash_list_regex, input_text)
    found_list = re.findall(dash_list_regex, input_text)

    # reformat unordered list with '-' symbol
    for i in range(len(split_list)):
        refactored_answer += split_list[i]
        if i < len(found_list):
            refactored_answer += found_list[i].replace(" -", "<br> - ")
            refactored_answer += "<br>"

    number_list_regex = " [0-9]+\..*?\. "
    split_list = re.split(number_list_regex, refactored_answer)
    found_list = re.findall(number_list_regex, refactored_answer)

    refactored_answer = ""

    for i in range(len(split_list)):
        refactored_answer += split_list[i]
        if i < len(found_list):
            list_items = re.split(" [0-9]+\.", found_list[i])

            # ignore first elements because it's an empty string
            for j in range(1, len(list_items)):
                refactored_answer += f"<br> {j}. "
                refactored_answer += list_items[j]

            refactored_answer += "<br>"

    return refactored_answer

dash_list_regex = re.compile(" -[^ ].*?\. ")
number_list_regex = re.compile(" [0-9]+\..*?\. ")

def is_response(email_subject):
    # This function now correctly checks if the email_subject string contains "Re:"
    return bool(re.search(r"\bRe:", email_subject, re.IGNORECASE)) if email_subject else False

# Function to handle SIGTERM signal
def sigterm_handler(signum, frame):
    global running
    print("Received SIGTERM. Exiting gracefully.")
    running = False

# Register the signal handler
signal.signal(signal.SIGTERM, sigterm_handler)


def continuous_process_emails():
    global running
    while running:  # Change to an infinite loop to keep checking indefinitely
        try:
            global error_encounter
            error_encounter = False  # Reset the flag at the beginning of each iteration
            
            qa_llm = create_qa_llm()
            print("Checking for new emails...")
            process_emails(qa_llm)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Instead of setting error_encounter to True, we log the error and continue

        for _ in range(2):  # Sleep for 2 seconds, but allow interruption
            time.sleep(1)
            if not running: # Check if we've received SIGTERM
                break

        if error_encounter:
            print("An error was encountered in the last iteration, but continuing to check emails.")
            
        print("Waiting before the next email check...")

if __name__ == "__main__":
    continuous_process_emails()