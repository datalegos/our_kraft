from flask import Flask, request, jsonify
from flask_cors import CORS
import tiktoken
import os
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token pricing (per 1K tokens) - OpenAI GPT-4 pricing as example
TOKEN_PRICING = {
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
    'claude-3-opus': {'input': 0.015, 'output': 0.075},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125}
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'tokenization-api'}), 200

@app.route('/api/tokens/count', methods=['POST'])
def count_tokens():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        model = data.get('model', 'gpt-4')
        
        # Get appropriate encoding for the model
        try:
            if model.startswith('gpt-4'):
                encoding = tiktoken.encoding_for_model('gpt-4')
            elif model.startswith('gpt-3.5'):
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            else:
                # Default to cl100k_base for most modern models
                encoding = tiktoken.get_encoding('cl100k_base')
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        
        # Count tokens
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        # Calculate character and word counts for reference
        char_count = len(text)
        word_count = len(text.split())
        
        return jsonify({
            'status': 'success',
            'tokenCount': token_count,
            'characterCount': char_count,
            'wordCount': word_count,
            'model': model,
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Token counting error: {str(e)}")
        return jsonify({'error': 'Failed to count tokens', 'details': str(e)}), 500

@app.route('/api/tokens/estimate-cost', methods=['POST'])
def estimate_cost():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        model = data.get('model', 'gpt-4')
        operation_type = data.get('type', 'input')  # 'input' or 'output'
        
        # Get token count
        try:
            if model.startswith('gpt-4'):
                encoding = tiktoken.encoding_for_model('gpt-4')
            elif model.startswith('gpt-3.5'):
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            else:
                encoding = tiktoken.get_encoding('cl100k_base')
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        
        tokens = encoding.encode(text)
        token_count = len(tokens)
        
        # Calculate cost
        if model in TOKEN_PRICING:
            price_per_1k = TOKEN_PRICING[model][operation_type]
            estimated_cost = (token_count / 1000) * price_per_1k
        else:
            # Default to GPT-4 pricing if model not found
            price_per_1k = TOKEN_PRICING['gpt-4'][operation_type]
            estimated_cost = (token_count / 1000) * price_per_1k
        
        return jsonify({
            'status': 'success',
            'tokenCount': token_count,
            'model': model,
            'operationType': operation_type,
            'pricePerThousandTokens': price_per_1k,
            'estimatedCost': round(estimated_cost, 6),
            'currency': 'USD',
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cost estimation error: {str(e)}")
        return jsonify({'error': 'Failed to estimate cost', 'details': str(e)}), 500

@app.route('/api/tokens/batch-count', methods=['POST'])
def batch_count_tokens():
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        model = data.get('model', 'gpt-4')
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be an array'}), 400
        
        # Get appropriate encoding
        try:
            if model.startswith('gpt-4'):
                encoding = tiktoken.encoding_for_model('gpt-4')
            elif model.startswith('gpt-3.5'):
                encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            else:
                encoding = tiktoken.get_encoding('cl100k_base')
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        
        results = []
        total_tokens = 0
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue
                
            tokens = encoding.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            
            results.append({
                'index': i,
                'tokenCount': token_count,
                'characterCount': len(text),
                'wordCount': len(text.split()),
                'text_preview': text[:50] + '...' if len(text) > 50 else text
            })
        
        return jsonify({
            'status': 'success',
            'totalTexts': len(texts),
            'totalTokens': total_tokens,
            'averageTokens': round(total_tokens / len(texts), 2) if texts else 0,
            'model': model,
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch token counting error: {str(e)}")
        return jsonify({'error': 'Failed to count tokens', 'details': str(e)}), 500

@app.route('/api/tokens/models', methods=['GET'])
def get_supported_models():
    return jsonify({
        'status': 'success',
        'supportedModels': list(TOKEN_PRICING.keys()),
        'pricing': TOKEN_PRICING,
        'currency': 'USD',
        'priceUnit': 'per 1K tokens',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)