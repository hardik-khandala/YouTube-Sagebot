# QNA Chatbot

## Overview

This project is an advanced NLP system designed for dynamic data extraction and question answering from YouTube video transcripts. It leverages the `youtube_transcript_api` for transcript retrieval and combines it with the generative capabilities of GPT-2 to provide comprehensive and contextually accurate responses.

## Features

- **Dynamic Data Extraction**: Utilizes `youtube_transcript_api` to fetch transcripts from YouTube videos.
- **Hybrid Model**: Combines contextual understanding from video transcripts with the generative power of GPT-2.
- **Enhanced Responses**: Transforms concise single-line answers into detailed and comprehensive elaborations.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/hardik-khandala/QNA-Chatbot.git
    ```

2. **Install Requirements**

    Ensure you have Python installed, then run the following command to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```