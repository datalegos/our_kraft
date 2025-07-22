from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import logging
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI configuration
openai.api_key = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')

# Tokenization API URL (internal service)
TOKENIZATION_API_URL = os.getenv('TOKENIZATION_API_URL', 'http://localhost:5002')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'summarization-api'}), 200

@app.route('/api/summarize/article', methods=['POST'])
def summarize_article():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Article content is required'}), 400
        
        content = data['content']
        summary_length = data.get('length', 'medium')  # short, medium, long
        model = data.get('model', 'gpt-3.5-turbo')
        
        # Define summary lengths
        length_instructions = {
            'short': 'in 1-2 sentences',
            'medium': 'in 3-4 sentences', 
            'long': 'in 5-7 sentences'
        }
        
        length_instruction = length_instructions.get(summary_length, 'in 3-4 sentences')
        
        # Create prompt
        prompt = f"""Please summarize the following news article {length_instruction}. Focus on the key facts, main points, and important details:

{content}

Summary:"""
        
        # Get token count for cost estimation
        token_response = requests.post(f"{TOKENIZATION_API_URL}/api/tokens/count", 
                                     json={'text': prompt, 'model': model})
        
        input_tokens = 0
        if token_response.status_code == 200:
            input_tokens = token_response.json().get('tokenCount', 0)
        
        # Generate summary using OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional news summarizer. Create clear, concise, and informative summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200 if summary_length == 'short' else 300 if summary_length == 'medium' else 400,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Get output token count
        output_token_response = requests.post(f"{TOKENIZATION_API_URL}/api/tokens/count",
                                            json={'text': summary, 'model': model})
        
        output_tokens = 0
        if output_token_response.status_code == 200:
            output_tokens = output_token_response.json().get('tokenCount', 0)
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'originalLength': len(content),
            'summaryLength': len(summary),
            'compressionRatio': round(len(summary) / len(content), 2),
            'model': model,
            'summaryType': summary_length,
            'tokenUsage': {
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'totalTokens': input_tokens + output_tokens
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'error': 'Failed to generate summary', 'details': str(e)}), 500
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/summarize/batch', methods=['POST'])
def summarize_batch():
    try:
        data = request.get_json()
        if not data or 'articles' not in data:
            return jsonify({'error': 'Articles array is required'}), 400
        
        articles = data['articles']
        summary_length = data.get('length', 'medium')
        model = data.get('model', 'gpt-3.5-turbo')
        
        if not isinstance(articles, list):
            return jsonify({'error': 'Articles must be an array'}), 400
        
        results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for i, article in enumerate(articles):
            if not isinstance(article, dict) or 'content' not in article:
                continue
            
            try:
                # Summarize individual article
                summary_response = requests.post('http://localhost:5003/api/summarize/article',
                                               json={
                                                   'content': article['content'],
                                                   'length': summary_length,
                                                   'model': model
                                               })
                
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    
                    result = {
                        'index': i,
                        'title': article.get('title', f'Article {i+1}'),
                        'summary': summary_data['summary'],
                        'originalLength': summary_data['originalLength'],
                        'summaryLength': summary_data['summaryLength'],
                        'compressionRatio': summary_data['compressionRatio'],
                        'tokenUsage': summary_data['tokenUsage']
                    }
                    
                    total_input_tokens += summary_data['tokenUsage']['inputTokens']
                    total_output_tokens += summary_data['tokenUsage']['outputTokens']
                    
                    results.append(result)
                else:
                    results.append({
                        'index': i,
                        'title': article.get('title', f'Article {i+1}'),
                        'error': 'Failed to summarize'
                    })
                    
            except Exception as e:
                logger.error(f"Error summarizing article {i}: {str(e)}")
                results.append({
                    'index': i,
                    'title': article.get('title', f'Article {i+1}'),
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'totalArticles': len(articles),
            'successfulSummaries': len([r for r in results if 'summary' in r]),
            'model': model,
            'summaryType': summary_length,
            'totalTokenUsage': {
                'inputTokens': total_input_tokens,
                'outputTokens': total_output_tokens,
                'totalTokens': total_input_tokens + total_output_tokens
            },
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch summarization error: {str(e)}")
        return jsonify({'error': 'Failed to process batch', 'details': str(e)}), 500

@app.route('/api/summarize/custom', methods=['POST'])
def custom_summarize():
    try:
        data = request.get_json()
        if not data or 'content' not in data or 'instructions' not in data:
            return jsonify({'error': 'Content and instructions are required'}), 400
        
        content = data['content']
        instructions = data['instructions']
        model = data.get('model', 'gpt-3.5-turbo')
        max_tokens = data.get('maxTokens', 300)
        
        # Create custom prompt
        prompt = f"""Following these specific instructions: {instructions}

Please process this news article content:

{content}

Result:"""
        
        # Get token count
        token_response = requests.post(f"{TOKENIZATION_API_URL}/api/tokens/count",
                                     json={'text': prompt, 'model': model})
        
        input_tokens = 0
        if token_response.status_code == 200:
            input_tokens = token_response.json().get('tokenCount', 0)
        
        # Generate custom summary
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that follows instructions precisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # Get output token count
        output_token_response = requests.post(f"{TOKENIZATION_API_URL}/api/tokens/count",
                                            json={'text': result, 'model': model})
        
        output_tokens = 0
        if output_token_response.status_code == 200:
            output_tokens = output_token_response.json().get('tokenCount', 0)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'instructions': instructions,
            'model': model,
            'tokenUsage': {
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'totalTokens': input_tokens + output_tokens
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return jsonify({'error': 'Failed to process content', 'details': str(e)}), 500
    except Exception as e:
        logger.error(f"Custom summarization error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)