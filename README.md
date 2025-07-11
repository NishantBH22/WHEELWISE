# WHEELWISE

A web application that helps users find cars based on their specifications, built with Flask, Tailwind CSS, and a SQL database. It uses natural language processing with fuzzy matching to parse user inputs and return top car matches, displayed in a dynamic table with CSV export functionality.

## Features
- User-friendly interface with Tailwind CSS styling
- Backend powered by Flask and SQL database
- Fuzzy matching for flexible input parsing
- Dynamic table with car matches and CSV download
- Responsive design with notifications

## Technologies Used
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
- **Backend**: Python, Flask, Pandas, SQLAlchemy, fuzzywuzzy
- **Database**: Microsoft SQL Server

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/NishantBH22/WHEELWISE.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up a SQL database named **CarSpecs** and update the connection string in `backend.py`.

4. Run the Flask server:
   ```bash
   python backend.py
   ```

5. Open [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

- Enter car specifications (e.g., **"Toyota sedan 2020 200 hp automatic"**) in the textarea.
- Click **"Find Cars"** to see matching cars in a dynamic table.
- Click **"Download Results as CSV"** to export the results.
- Click **"Clear"** to reset the input and results.
