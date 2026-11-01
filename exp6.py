
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample e-commerce review webpage
url = "https://webscraper.io/test-sites/e-commerce/static"

# Fetch webpage
response = requests.get(url, timeout=10)
response.raise_for_status()

# Parse HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Extract product descriptions as sample review text
reviews = [
    " ".join(item.stripped_strings)
    for item in soup.select(".description")
]

# Use sample reviews if the page has no matching content
if not reviews:
    reviews = [
        "Excellent product with good quality",
        "Amazing product and excellent performance",
        "Good quality product at a reasonable price",
        "Excellent performance and easy to use"
    ]

# Combine reviews
text = " ".join(reviews)

# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Product Review Word Cloud")
plt.show()
