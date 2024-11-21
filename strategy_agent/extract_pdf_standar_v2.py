from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import os
import re
import os
import base64
from openai import OpenAI
from datetime import datetime  # Ensure datetime is imported

########################################################### preprocess ##########################################################

# Configuration for processing based on keyword
CATEGORY_CONFIG = {
    "PASSES": {
        "parts_to_capture": ["top_half"],  # Requires splitting into top and bottom halves
        "page_range": (12, 18),  # Specify the page range for "PASSES"
    },
    "PlayerTimeShot": {
        "parts_to_capture": ["whole_page"],  # Requires the whole page
        "page_range": (11, 16),  # Specify the page range for "SHOTS"
    },
    "DUELS": {
        "parts_to_capture": ["top_half"],  # Requires only the top half
        "page_range": (14, 18),  # Specify the page range for "DUELS"
    },
    "LOSSES": {
        "parts_to_capture": ["top_half", "bottom_half"],  # Default: save the whole page
        "page_range": (15, 20),  # General fallback range
    },
}
OUTPUT_DATE_FORMAT = "%Y-%m-%d"  # Modify as needed

def find_passing_pages(pdf_path, keyword="PASSES", page_range=(10, 18)):
    """
    Identify all pages containing the keyword with exact capitalization within the specified page range.
    """
    reader = PdfReader(pdf_path)
    passing_pages = []
    start, end = page_range
    for i in range(start - 1, end):  # Convert 1-based to 0-based indexing
        text = reader.pages[i].extract_text()
        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
          return_date = dates[0]
        if keyword in text:  # Case-sensitive search
            passing_pages.append(i + 1)  # Return 1-based page numbers
    return passing_pages, return_date

def find_pages_with_keyword(pdf_path, keyword, page_range):
    """
    Identify all pages containing the keyword with exact capitalization within the specified page range.
    Returns a list of tuples: (page_number, date).
    """
    reader = PdfReader(pdf_path)
    pages_with_keyword = []
    start, end = page_range
    for i in range(start - 1, end):  # Convert 1-based to 0-based indexing
        text = reader.pages[i].extract_text()
        date_pattern = r'\b\d{2}\.\d{2}\.\d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
          return_date = dates[0]
        if keyword in text:  # Case-sensitive search
            if not return_date:
              return_date = 'unknown_date'
            pages_with_keyword.append((i + 1, return_date))
    return pages_with_keyword

def generate_unique_filename(base_path, base_name, extension, sequence_counter=1):
    """
    Generate a unique filename by appending a sequence number if a file already exists.
    """
    unique_path = f"{base_path}/{base_name}{extension}"
    while os.path.exists(unique_path):
        unique_path = f"{base_path}/{base_name}_{sequence_counter}{extension}"
        sequence_counter += 1
    return unique_path

def screenshot_page_parts(
    pdf_path,
    found_pages,
    output_dir,
    school_names,
    keyword,
    processed_pages
):
    """
    Take screenshots of the specified parts of the pages based on keyword logic and save them with unique names.
    Handles `found_pages` as a list of tuples [(page_number, date)].
    """
    def find_consecutive_pages(pages):
        """
        Helper function to find the first two consecutive pages in the list.
        """
        for i in range(len(pages) - 1):
            current_page, _ = pages[i]
            next_page, _ = pages[i + 1]
            if next_page - current_page == 1:  # Check if pages are consecutive
                return [pages[i], pages[i + 1]]
        return []

    # Define processing logic for each keyword
    if keyword == "PASSES":
        if len(found_pages) < 2:
            print(f"Keyword 'PASSES' requires at least 2 consecutive pages. Skipping.")
            return
        pages_to_process = find_consecutive_pages(found_pages)
        if not pages_to_process:
            print(f"No two consecutive pages found for 'PASSES'. Skipping.")
            return
        parts_to_capture = ['top_half']
    elif keyword == "PlayerTimeShot":
        if len(found_pages) < 2:
            print(f"Keyword 'SHOTS' requires at least 2 consecutive pages. Skipping.")
            return
        pages_to_process = find_consecutive_pages(found_pages)
        if not pages_to_process:
            print(f"No two consecutive pages found for 'SHOTS'. Skipping.")
            return
        parts_to_capture = ['whole_page']
    elif keyword == "DUELS":
        pages_to_process = [found_pages[0]]  # Single page for DUELS
        parts_to_capture = ['top_half']
    elif keyword == "LOSSES":
        pages_to_process = [found_pages[0]]  # Single page for LOSS
        parts_to_capture = ['top_half', 'bottom_half']
    else:
        print(f"Keyword '{keyword}' is not configured. Skipping.")
        return

    print(f"Processing pages: {pages_to_process}")

    for i, (page_number, date) in enumerate(pages_to_process):
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if not images:
            print(f"Failed to render page {page_number}.")
            continue
        page_image = images[0]
        width, height = page_image.size

        # Handle school naming logic based on keyword
        if keyword in ["PASSES", "PlayerTimeShot"]:
            school_name = school_names[i]  # First page for first school, second for second school
        elif keyword == "DUELS":
            school_name = f"{school_names[0]}_{school_names[1]}"  # Both schools for DUELS
        elif keyword == "LOSSES":
            school_name = school_names[0] if parts_to_capture[0] == 'top_half' else school_names[1]
        else:
            continue

        for part in parts_to_capture:
            # Adjust school name for LOSS
            if keyword == "LOSSES" and part == "bottom_half":
                school_name = school_names[1]

            # Generate unique ID and filename
            school_name_cleaned = school_name.replace(" ", "_").replace("/", "_")
            date_suffix = f"_{date}" if date else ""
            unique_id = f"{school_name_cleaned}_{date_suffix}_{part}_{keyword}_{page_number}"

            if unique_id in processed_pages:  # Avoid redundancy
                print(f"Skipping duplicate processing for {unique_id}.")
                continue

            # Determine the cropping region based on the part to capture
            if part == 'top_half':
                cropped_image = page_image.crop((0, 0, width, height // 2))
            elif part == 'bottom_half':
                cropped_image = page_image.crop((0, height // 2, width, height))
            elif part == 'whole_page':
                cropped_image = page_image  # No cropping for the whole page
            else:
                print(f"Unknown part '{part}'. Skipping.")
                continue

            # Save the image
            base_name = f"{school_name_cleaned}{date_suffix}_{part}_{keyword}"
            extension = ".jpg"
            output_path = generate_unique_filename(output_dir, base_name, extension)
            cropped_image.save(output_path)
            print(f"Screenshot saved to {output_path}")
            processed_pages.add(unique_id)  # Mark as processed


def extract_school_names_from_filename(pdf_path):
    """
    Extract the school names from the PDF file name.
    """
    filename = os.path.basename(pdf_path).replace(".pdf", "")
    parts = filename.split(" - ")
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()  # Return the first two school names
    return "School_1", "School_2"  # Default names if format is unexpected

def process_all_pdfs(input_dir, output_dir, keyword):
    """
    Process all PDF files in the input directory, identify pages with the keyword,
    and save screenshots with extracted school names and unique filenames.
    """
    # Fetch configuration for the specified keyword
    config = CATEGORY_CONFIG.get(keyword)
    parts_to_capture = config["parts_to_capture"]
    page_range = config["page_range"]

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    processed_pages = set()  # Track processed (pdf, page, part)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")

            # Step 1: Identify pages with keyword
            pages_with_keyword = find_pages_with_keyword(pdf_path, keyword, page_range)
            if pages_with_keyword:
                print(f"Pages with keyword '{keyword}' found: {pages_with_keyword}")

                # Step 2: Extract school names
                school_names = extract_school_names_from_filename(pdf_path)
                print(f"School names extracted: {school_names}")

                # Step 3: Screenshot specified parts
                screenshot_page_parts(pdf_path, pages_with_keyword, output_dir, school_names, keyword, processed_pages)
            else:
                print(f"No pages with keyword '{keyword}' found in {filename}.")

########################################################### preprocess ##########################################################

########################################################### Chatbot  ############################################################

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process images and query the OpenAI API
def process_and_query_images(client, school, folder_path, prompt_text, n=5):
    # Collect all images with the school's name in the filename
    print(f"Searching for images related to {school}...")
    school_images = [f for f in os.listdir(folder_path) if school in f and f.endswith((".jpg", ".png"))]

    if not school_images:
        print(f"No images with '{school}' found in {folder_path}.")
        return None

    # Extract dates from filenames and sort by date
    date_format = "%d.%m.%Y"  # Assuming filenames contain dates in this format
    image_dates = []

    for image in school_images:
        try:
            # Extract the date part of the filename
            parts = image.split("_")
            for part in parts:
                if "." in part:  # Likely a date part
                    image_date = datetime.strptime(part, date_format)
                    image_dates.append((image, image_date))
                    break
        except Exception as e:
            print(f"Could not parse date from filename '{image}': {e}")

    if not image_dates:
        print(f"No valid dates found in filenames for '{school}'.")
        return []

    # Sort by date (most recent first)
    image_dates.sort(key=lambda x: x[1], reverse=True)

    # Get the closest n images
    closest_images = image_dates[:n]

    # Encode each image as base64 and create the message payload
    image_messages = []
    for image_filename, _ in closest_images:
        image_path = os.path.join(folder_path, image_filename)
        base64_image = encode_image(image_path)
        image_messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            }
        )

    # Prepare the query with the images
    query_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text.strip(),
                },
                *image_messages,  # Append all image messages
            ],
        }
    ]

    # Send the query to the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=query_message,
        max_tokens=500,
    )

    # Return the response content
    return response.choices[0].message.content


########################################################### Chatbot  ############################################################