# RAG Assistant

A Streamlit PDF question-answering app using Chroma, Hugging Face embeddings, and Mistral.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your Mistral API key.
4. Start the app:

   ```powershell
   streamlit run rag_app.py
   ```

The checked-in `chroma-db/` directory contains the initial vector database. Uploaded PDFs are added to the running app's local database.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. Keep `.env` out of the repository.
2. In Streamlit Community Cloud, choose **Create app** and select this repository and branch.
3. Set **Main file path** to `rag_app.py`.
4. Open **Advanced settings > Secrets** and add:

   ```toml
   MISTRAL_API_KEY = "your_mistral_api_key"
   ```

5. Deploy. Community Cloud installs the packages from `requirements.txt`.

The app must keep `chroma-db/` in the repository unless you change the application to build the database during deployment. Streamlit Community Cloud storage is ephemeral, so PDFs uploaded while the app is running are not a permanent backup.
