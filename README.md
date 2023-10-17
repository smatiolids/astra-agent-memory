# Astra Agent Memory with PDF context

![Astra Agent Memory](astra_agent.png)

The purpose of this demo is to combine the processing of PDF files, embedding generation, multiple retrieval metrics and a user interface with streamlit that also consider agent memory.

## Installing dependencies

pip install -r requirements.txt

## DataStax Astra

Create an account and a Vector DB at (astra.datastax.com).

## Environment Variables

Define the AstraDB credentials and Open AI API Key in the .env file.

Copy .env.sample to .env

## Running

streamlit run app.py

## Loading PDF

I uploaded and converted PDF using the notebook "Explicando Retrieval Augmented Generation.ipynb". 
