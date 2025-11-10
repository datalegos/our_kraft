from flask import Flask, render_template, request, jsonify
import stripe
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Stripe with keys from environment
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
stripe_publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')

@app.route('/')
def index():
    """Render the main payment page"""
    return render_template('index.html', 
                         stripe_publishable_key=stripe_publishable_key)

# ============= STRIPE ENDPOINTS =============

@app.route('/api/stripe/create-payment-intent', methods=['POST'])
def create_payment_intent():
    """Create a Stripe Payment Intent"""
    try:
        data = request.get_json()
        amount = data.get('amount')  # Amount in dollars/rupees
        currency = data.get('currency', 'usd').lower()
        
        # Stripe expects amount in smallest currency unit (cents for USD, paise for INR)
        amount_in_cents = int(float(amount) * 100)
        
        # Validate minimum amount
        min_amounts = {
            'usd': 50,  # $0.50
            'eur': 50,  # €0.50
            'inr': 5000  # ₹50.00
        }
        
        min_amount = min_amounts.get(currency, 50)
        if amount_in_cents < min_amount:
            min_display = min_amount / 100
            return jsonify({
                'error': f'Amount must be at least {min_display} {currency.upper()}'
            }), 400
        
        # Create Payment Intent
        intent = stripe.PaymentIntent.create(
            amount=amount_in_cents,
            currency=currency,
            automatic_payment_methods={'enabled': True},
            metadata={
                'integration': 'payment_gateway_poc'
            }
        )
        
        return jsonify({
            'clientSecret': intent.client_secret,
            'paymentIntentId': intent.id,
            'amount': intent.amount,
            'currency': intent.currency,
            'status': intent.status
        }), 200
        
    except Exception as e:
        # Handle all Stripe errors
        error_message = str(e)
        if 'stripe' in error_message.lower():
            return jsonify({'error': error_message}), 400
        return jsonify({'error': str(e)}), 500

@app.route('/api/stripe/retrieve-payment-intent/<intent_id>', methods=['GET'])
def retrieve_payment_intent(intent_id):
    """Retrieve Stripe Payment Intent details"""
    try:
        intent = stripe.PaymentIntent.retrieve(intent_id)
        
        return jsonify({
            'id': intent.id,
            'status': intent.status,
            'amount': intent.amount,
            'currency': intent.currency,
            'created': intent.created,
            'payment_method': intent.payment_method
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stripe/confirm-payment', methods=['POST'])
def confirm_payment():
    """Confirm a Stripe payment (webhook simulation)"""
    try:
        data = request.get_json()
        payment_intent_id = data.get('payment_intent_id')
        
        # Retrieve the payment intent
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        return jsonify({
            'status': 'success',
            'payment_intent_id': intent.id,
            'payment_status': intent.status,
            'amount': intent.amount,
            'currency': intent.currency,
            'message': 'Payment confirmed successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Payment Gateway POC - Stripe Integration")
    print("="*60)
    
    if stripe.api_key and stripe.api_key.startswith('sk_test_'):
        print("\n✅ Stripe configured (Test Mode)")
    else:
        print("\n⚠️  Stripe not configured")
        print("   Add your keys to .env file")
    
    print(f"\n🌐 Server running at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
