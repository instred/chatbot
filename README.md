# Mailbot - Automated Mail Response Chatbot

Mailbot is a chatbot designed to automatically respond to user emails. It utilizes a combination of web scraping techniques to gather incoming emails, callback functions in PHP, and the power of the Mistral model 7B for natural language understanding. Additionally, it integrates the Google Translate API to handle translations for multilingual support.

## Features

- **Automated Email Response:** Mailbot is capable of parsing incoming emails and generating appropriate responses based on the content of the email. It leverages question-answer pairs to get the context.
- **Web Scraping:** Utilizes callback functions in PHP to scrape incoming emails from a designated inbox.
- **Natural Language Understanding:** Powered by the Mistral model 7B, Mailbot can understand and generate human-like responses to user inquiries.
- **Multilingual Support:** Integrates the Google Translate API to handle translation of emails, enabling communication in multiple languages. An API Key is needed for it to work.


## Quick recap on how it works

1. **Receiving Emails:** Mailbot continuously monitors the designated email inbox for incoming messages using web scraping techniques implemented with callback functions in PHP.
2. **Processing Incoming Emails:** Upon receiving a new email, Mailbot processes the JSON request and extracts all relevant data, including the email ID, sender information, and message content.
3. **Filtering Automated Replies:** Mailbot checks if the incoming email is a reply or an automated response to avoid unnecessary processing. If it detects that the email is not a user-generated response, it skips processing and moves to the next email.
4. **Language Detection:** If the email is not an automated reply, Mailbot uses spaCy to determine if the email is written in English. If it detects that the email is in English, Mailbot proceeds to generate a response using the Mistral model.
5. **Translation for Non-English Emails:** If the email is not in English, Mailbot utilizes the Google Translate API to translate the email content into English before generating a response. This ensures that Mailbot can communicate effectively with users in multiple languages.
6. **Refactoring and Pretty Formatting:** Mailbot undergoes small refactoring processes to enhance code readability and maintainability, ensuring that it remains easy to understand and modify.
7. **Sending Responses:** After processing the email and generating a response, Mailbot uses PHP to send the response back to the original sender, completing the automated email response process seamlessly.

This workflow allows Mailbot to efficiently handle incoming emails, provide appropriate responses, and facilitate effective communication with users through automated email interactions.


## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/mailbot.git
