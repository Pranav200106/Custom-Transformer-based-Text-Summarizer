# Custom Transformer-based Text Summarizer

This project implements a custom Transformer architecture in PyTorch for abstractive text summarization. It features a full-stack application with a Flask backend and a React frontend, enabling real-time text summarization through a user-friendly web interface.

## Features

*   **Custom Transformer Architecture:** Designed and implemented an encoder-decoder Transformer model from scratch using PyTorch, incorporating multi-head attention mechanisms and positional encoding.
*   **Abstractive Text Summarization:** Generates concise and coherent summaries that may include words not present in the original text.
*   **Full-Stack Application:**
    *   **Backend (Flask):** Provides an API endpoint for text summarization, handling model inference and data processing.
    *   **Frontend (React):** Offers an intuitive web interface for users to input text and receive summaries in real-time.
*   **Real-time Summarization:** Experience immediate summarization results as you type or paste text.
*   **Model Training & Inference:** Includes scripts for training the Transformer model and performing inference.

## Project Structure

```
.
├── LICENSE
├── README.md
├── backend/
│   ├── app.py                  # Flask application entry point
│   ├── Pipfile                 # Python dependency management
│   ├── Pipfile.lock
│   ├── train_model.py          # Script for training the Transformer model
│   └── model/
│       ├── __init__.py
│       ├── inference.py        # Model inference logic
│       ├── training.py         # Model training utilities
│       ├── transformer.py      # Custom Transformer model definition
│       └── checkpoints/
│           └── best_model.pth  # Pre-trained model checkpoint
└── frontend/
    ├── .gitignore
    ├── eslint.config.js
    ├── index.html              # Frontend entry HTML
    ├── package-lock.json
    ├── package.json            # Frontend dependencies
    ├── vite.config.js
    └── src/
        ├── app.css
        ├── App.jsx             # Main React application component
        ├── index.css
        └── main.jsx            # React application entry point
```

## Installation

To set up the project locally, follow these steps:

### Prerequisites

*   Python 3.8+
*   Node.js (LTS recommended)
*   pipenv (for Python dependency management)
*   npm or yarn (for Node.js package management)

### Backend Setup

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install Python dependencies using Pipenv:
    ```bash
    pipenv install
    ```
3.  Activate the Pipenv shell:
    ```bash
    pipenv shell
    ```
4.  (Optional) Train the model:
    ```bash
    python train_model.py
    ```
    A pre-trained model checkpoint (`best_model.pth`) is provided in `backend/model/checkpoints/`.
5.  Run the Flask backend server:
    ```bash
    python app.py
    ```
    The backend server will typically run on `http://127.0.0.1:5000`.

### Frontend Setup

1.  Open a new terminal and navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install Node.js dependencies:
    ```bash
    npm install
    # or yarn install
    ```
3.  Start the React development server:
    ```bash
    npm run dev
    # or yarn dev
    ```
    The frontend application will typically run on `http://localhost:5173`.

## Usage

1.  Ensure both the Flask backend and React frontend servers are running.
2.  Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`).
3.  Enter or paste the text you wish to summarize into the provided input area.
4.  The summarized text will appear in real-time.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the [LICENSE](LICENSE) file.
