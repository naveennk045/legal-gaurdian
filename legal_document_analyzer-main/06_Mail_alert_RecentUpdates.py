import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq


API_KEY = "Your apikey"
MODEL_NAME = "llama-3.2-11b-vision-preview"
llm = ChatGroq(groq_api_key=API_KEY, model_name=MODEL_NAME)



#===========================================================================================================================================================
#===========================================================================================================================================================

def scrape_legal_news(url):
    print("Scrapping content ------------>>\n\n")
    response = requests.get(url)
    response.raise_for_status() 
    soup = BeautifulSoup(response.content, 'html.parser')

    # Adjust selectors to target titles and links
    headlines = soup.select('h1, h2, h3, div')  # Update with appropriate selectors
    articles = []
    for headline in headlines:
        title = headline.get_text(strip=True) 
        link_tag = headline.find('a') 
        link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else None  
        if title:  # Ensure title is not empty
            articles.append({"title": title, "link": link or "No link available"})
    return articles



#===========================================================================================================================================================
#===========================================================================================================================================================


def summarize_news_in_chunks(news_list, max_chunk_size=7000):
 
    if not news_list:
        return "No news articles to summarize."

    chunks = []
    current_chunk = ""
    chunk_size = 0

    # Split the news into chunks based on max token size
    for i, news in enumerate(news_list, 1):
        title = news.get('title', 'No title available')
        link = news.get('link', 'No link available')
        news_item = f"{i}. {title} ({link})\n"
        token_count = len(news_item)  # Approximate token count using character length

        # If adding this news exceeds the chunk size, start a new chunk
        if chunk_size + token_count > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = news_item
            chunk_size = token_count
        else:
            current_chunk += news_item
            chunk_size += token_count

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Summarize each chunk using the Llama model
    print("Model call...\n\n")

    summaries = []
    for chunk in chunks:
        input_text = f"Summarize the following legal updates into key sentences: more simpler in 2 lines \n{chunk}"
        response = llm.invoke(input_text)  # Call Llama model
        summaries.append(response.content)

    # Combine all summaries into one
    return "\n\n".join(summaries)



#===========================================================================================================================================================
#===========================================================================================================================================================


# Step 3: Send the email
def send_email(subject, body, recipient_email, sender_email, sender_password):
    print("Sending Mail...\n\n")
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    # Connect to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())





#===========================================================================================================================================================
#===========================================================================================================================================================





def main():
    # legal_news_url = input()
    legal_news_url = "https://www.freelaw.in/legalarticles/Protecting-the-Rights-of-Workers-in-India-" 
    recipient_email = "nandhiraja16@gmail.com"
    sender_email = "nandhiraja16@gmail.com"
    sender_password = "sxed rmzk cyui risw"


    # Scrape and summarize news
    scraped_news = scrape_legal_news(legal_news_url)
    if not scraped_news:
        print("No news found.")
        return


    summarized_news = summarize_news_in_chunks(scraped_news)

    # Create email body with alignment
    email_body = "<html><body>"
    email_body += "<h2>Recent Legal Updates</h2>"
    email_body += "<ul>"
    for point in summarized_news.split('\n'):
        email_body += f"<li>{point.strip()}</li>"
    email_body += "</ul>"
    email_body += "</body></html>"



    # Send email
    send_email(
        subject="Important Legal Updates",
        body=email_body,
        recipient_email=recipient_email,
        sender_email=sender_email,
        sender_password=sender_password
    )

    print("Email sent successfully!")




if __name__ == "__main__":
    main()
