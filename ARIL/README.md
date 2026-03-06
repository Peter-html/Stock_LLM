# ARIL - Stock Advisor

An intelligent stock analysis application powered by LLMs, ChromaDB, and real-time data processing.

## Features

- 📊 **Stock Fundamentals Analysis** - PE ratio, ROE, EPS analysis
- 📰 **Sentiment Analysis** - News sentiment tracking for stocks
- 📈 **Price History** - Historical price data analysis
- 🤖 **AI-Powered Insights** - Powered by Google GenAI
- 🔍 **Vector Search** - Fast semantic search using ChromaDB and sentence transformers

## Tech Stack

- **Backend**: Python
- **UI**: Streamlit
- **Vector DB**: ChromaDB
- **LLM**: Google GenAI
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Data Processing**: Pandas

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/Stock_LLM.git
cd Stock_LLM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google GenAI API key:
```bash
# Create a .env file and add:
GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
ARIL/
├── app.py                    # Main Streamlit application
├── pdf_parser.py             # PDF parsing utilities
├── inspect_pdf.py            # PDF inspection tools
├── requirements.txt          # Python dependencies
├── dataset/                  # Stock datasets
│   ├── stocks_fundamentals.csv
│   ├── stocks_news_sentiment.csv
│   └── stocks_price_history.csv
└── chroma_stock_db/          # ChromaDB vector database
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and suggestions, please open an GitHub issue.
