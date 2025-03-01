# find-magnifier
tool to locate magnifier and extract text following the symbol

## problem statement
There are some books in PDF format, where some pages have magnifier symbols in the margin. 
We want to 
- 1) identify if the page has at least one magnifier symbol 
- 2) extract the text following the symbol

## proposed workflow to handle the problem
- read in the pdf files page by page
- for each page
    - turn pdf page into a png image, save to `data/{filename}/images/{page_number}.png`
    - send the image into a strong vision model ( qwen2.5-VL-70b or o1 ) to check if it contains a magnifier
        - if no, skip   
        - if yes, send the picture into gpt to extract the following info: 
            - page id: int
            - cycle id: int
            - page number: int | str | None. note that initially the page number is roman numerals.
            - text after the symbol: str
- aggregate the results into a dataframe and save to csv

## Streamlit App Requirements

The final application will be deployed as a user-friendly Streamlit web app for non-technical content team members.

Input:
- PDF file upload functionality
- File size limit and supported format validation
- Preview of the uploaded PDF
- display total page number in the pdf file

Processing:
- Real-time progress bar showing overall completion
- Display of current page being processed
- Real-time display of extracted text and metadata if any
- Running tally of pages processed and magnifiers found

Output:
- Interactive preview of the extracted data in tabular format
- Download options:
  - CSV file with extracted data
- Success/failure notification

Error Handling:
- Graceful handling of invalid files
- Clear error messages for users
- Option to retry failed pages
- Automatic error logging for debugging

Additional Features:
- Save processing history
- Batch upload capability
- Basic data visualization of results

## tools
- langsmith for tracing and evaluating the model
- o1 or qwen2.5-VL-70b or gemini 2.0 flash for detection
- gpt-4o with structured output for extraction

## data
the data is stored in the `data` folder; use cycle 1 ( 34 pages ) for testing. 


## how to use 
```bash
# Create new conda environment with Python 3.10
conda create -n myenv python=3.10 --yes
# Activate the environment
conda activate myenv

# Install packages
pip install -r requirements.txt

# Run the Streamlit app
python3 -m streamlit run src/app.py
```

```bash
pytest src/tests/ --cov=src --cov-report=term-missing
```
