# Astra Agent Memory with PDF context

![Astra Agent Memory](astra_agent.png)

The purpose of this demo is to combine the processing of PDF files, embedding generation, multiple retrieval metrics and a user interface with streamlit that also consider agent memory.

I will use this demo at a Banking event.

Next step, is to bring caching

## Installing dependencies

pip install -r requirements.txt

## Environment Variables

Define the AstraDB credentials and Open AI API Key in the .env file.

Copy .env.sample to .env

## Running

streamlit run app.py

## Loading PDF

I uploaded and converted PDF using the notebook "Loading PDFs.ipynb". 
