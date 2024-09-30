from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import csv
import pandas as pd
import re
from langchain_openai import ChatOpenAI
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

# URL of the Twitter post
date = 'Jun21'
post_url = 'xxxxx'


# Function to convert cookie string to dictionary
def cookie_to_dic(cookie_string):
    cookies = {}
    parts = cookie_string.split('; ')
    for part in parts:
        key, value = part.split('=', 1)
        cookies[key] = value
    return cookies

# Path to your ChromeDriver executable
chrome_driver_path = '/Users/Path'  # Update Path with your actual path

#Save my data to ---
csv_file_path = f'/Users/Path/comments_data_{date}.csv'

# Cookie string
cookie_string = 'auth_token=xxxx; ct0=xxxxx'

# Convert cookie string to dictionary
cookies = cookie_to_dic(cookie_string)

# Set up the WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Open Twitter profile page
driver.get(post_url)

# Add cookies to the current session
for cookie in cookies:
    driver.add_cookie({
        'name': cookie,
        'value': cookies[cookie],
        'domain': '.x.com',  # Adjust domain as per your requirement
        'path': '/',
        'secure': True,
        'httpOnly': True
    })

# Refresh the page after adding cookies
driver.refresh()
time.sleep(5)

try:
    # Extract the main tweet content
    tweet_content = driver.find_element(By.XPATH, '//*[@data-testid="tweetText"]').text
    
    # Scroll and extract all comments, usernames, and timestamps
    last_height = driver.execute_script("return document.body.scrollHeight")

    comments_data = [] 
    
  # Add the main tweet content as a row with null values for other fields
    comments_data.append(['', '', '', '', tweet_content])
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)  # Increase time to allow comments to load

        # Extract comments, usernames, and their timestamps
        comment_elements = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        for comment in comment_elements:
            comment_text = comment.text.split('\n')
            if len(comment_text) >= 3:
                username = comment_text[0]
                user_handle = comment_text[1]
                timestamp = comment_text[3]
                content = comment_text[4]

                comments_data.append([username, user_handle, timestamp, content, ''])
                print(username, user_handle, timestamp, content)
        
        # Check if we have reached the bottom of the page
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


    # Write tweet content and comments data to CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Username', 'User Handle', 'Timestamp', 'Content', 'Tweet Content'])
        writer.writerows(comments_data)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the WebDriver
    driver.quit()


###Step2

# Load the data
comments_df = pd.read_csv(f'/Users/miyu/Desktop/comment/comments_data_{date}.csv', encoding='utf-8-sig')

def replace_nonsensical_data(content):
    if isinstance(content, str):
        # Remove URLs
        content = re.sub(r'https?://\S+', '', content)
        # Remove any substring that has no whitespace and is more than 40 characters long
        content = re.sub(r'\b\S{30,}\b', '', content)
        # Check if the content is purely numeric
        if re.fullmatch(r'\d+', content):
            return ''
        return content.strip()  # Remove leading and trailing whitespace
    return content


# Apply the function to the Content column
# comments_df['Content'] = comments_df['Content'].apply(delete_nonsensical_data)
comments_df['Content'] = comments_df['Content'].apply(replace_nonsensical_data)

# Drop rows with null or empty content
comments_df = comments_df[comments_df['Content'].str.strip() != '']

# Drop duplicate content, keeping only the first occurrence
comments_df = comments_df.drop_duplicates(subset=['Content'], keep='first')

content_column = 'Content'

            
# Function to translate text with retries
def translate_text_with_retry(text, to_lang='en', max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            translation = GoogleTranslator(source='auto', target=to_lang).translate(text)
            return translation
        except Exception as e:
            print(f"Error translating text: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying translation (attempt {attempt})...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to translate after {max_retries} attempts.")
                return ""

# Iterate through each row and translate non-English content
for index, row in comments_df.iterrows():
    try:
        # Check if the text needs translation (you may want to modify this condition)
        if not row[content_column].isascii():
            translated_text = translate_text_with_retry(row[content_column])
            comments_df.at[index, content_column] = translated_text
            print(f"Translated row {index}: {row[content_column]} => {translated_text}")
    except Exception as e:
        print(f"Error translating row {index}: {e}")


# Save the cleaned DataFrame (optional)
comments_df.to_csv(f'/Users/miyu/Desktop/comment/cleaned_comments_data_{date}.csv', index=False, encoding='utf-8-sig')

##Step3
# Initialize the GPT-4 model
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key="xxxxx")

# Function to read CSV file and return data
def read_csv(file_path):
    data = pd.read_csv(file_path)
    # Ensure all content entries are strings and drop any rows with NaN content
    data['Content'] = data['Content'].astype(str).fillna('')
    return data

# Function to save answers to a text file
def save_answers_to_file(answers, file_name):
    with open(file_name, 'w') as file:
        for answer in answers:
            file.write(answer + "\n\n")  # Adding new lines for readability

# Function to analyze sentiment and identify key elements
def analyze_survey_responses(data):
    comments = data['Content']
    
    # Perform sentiment analysis
    sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
    overall_sentiment = "Positive" if sum(sentiments) / len(sentiments) > 0 else "Negative"
    
    # Identify key elements using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', max_features=3)
    X = vectorizer.fit_transform(comments)
    key_elements = vectorizer.get_feature_names_out()
    
    return overall_sentiment, key_elements


def generate_explanations(key_elements, comments):
    explanations = []
    # Create a prompt to generate an explanation
    prompt = f"Based on the customer feedback provided, identify and explain the key topics that are being discussed the most. Please provide a list of these topics along with a brief explanation for each one in the context of the feedback. Key element: '{key_elements}'\n\nCustomer feedback:\n{comments}"
    response = llm(prompt)
    explanations.append(response.content.strip())
    return explanations


# Function to create and save a word cloud
def create_word_cloud(text, file_name):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(file_name, format='png')
    plt.close()

# Main function
def main(file_path, output_file):
    # Read CSV data
    data = read_csv(file_path)
    
    # Analyze survey responses
    overall_sentiment, key_elements = analyze_survey_responses(data)

    comments_text = " ".join(comment for comment in data['Content'])
    # Generate explanations for each key element
    explanations = generate_explanations(key_elements, comments_text)
    
    # Create answers
    answers = [
        f"Overall Sentiment: {overall_sentiment}",
        "Key Elements Mentioned:",
        *key_elements,
        "\nTopics Most Discussed:",
        *explanations
    ]
    
    # Save answers to a text file
    save_answers_to_file(answers, output_file)
    
    # Print answers to console with improved readability
    for answer in answers:
        print(answer + "\n")
    
    # Create and save word cloud
    create_word_cloud(comments_text, wordcloud_file)

# Specify the input CSV file path and output text file name
csv_file_path = f'/Users/miyu/Desktop/comment/cleaned_comments_data_{date}.csv'
output_text_file = f'/Users/miyu/Desktop/comment/answers_{date}.txt'
wordcloud_file = f'/Users/miyu/Desktop/comment/wordcloud_{date}.png'

# Call the main function
main(csv_file_path, output_text_file)

