import requests
from bs4 import BeautifulSoup
import json
import os

# URL of the free courses page
URL = "https://courses.analyticsvidhya.com/collections/courses"

# Output file path for saving scraped data
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/courses.json")


def scrape_courses():
    # Send GET request to the URL
    response = requests.get(URL)
    print(f"Response status: {response.status_code}")
    
    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage. Status code: {response.status_code}")

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Initialize list to hold course data
    courses = []

    # Locate course containers
    course_items = soup.find_all("li", class_="products__list-item")
    print(f"Found {len(course_items)} course containers.")

    # Loop through each course container to extract details
    for item in course_items:
        # Extract course link
        link_tag = item.find("a", class_="course-card")
        course_link = link_tag.get("href", "#") if link_tag else "#"
        if not course_link.startswith("http"):
            course_link = f"https://courses.analyticsvidhya.com{course_link}"

        # Extract course title
        title_tag = link_tag.find("h3") if link_tag else None
        title = title_tag.text.strip() if title_tag else "No Title"

        # Extract course image
        image_tag = link_tag.find("img", class_="course-card__img") if link_tag else None
        image_url = image_tag.get("src", "No Image URL") if image_tag else "No Image URL"

        # Extract course description
        lesson_tag = link_tag.find("span", class_="course-card__lesson-count") if link_tag else None
        description = lesson_tag.text.strip() if lesson_tag else "No Description"

        # Add the extracted details to the list
        courses.append({
            "title": title,
            "description": description,
            "image_url": image_url,
            "course_link": course_link,
        })

    # Debugging: Print the first few courses
    print(f"Scraped {len(courses)} courses.")
    for course in courses[:3]:
        print(course)

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save the course data to a JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(courses, f, indent=4)

    print(f"Data saved to {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    scrape_courses()
